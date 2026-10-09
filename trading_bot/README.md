# Fail-fast evolutionary trading bot

**Run it 24/7 on a server: [DEPLOY.md](DEPLOY.md) · On your Windows PC: [WINDOWS.md](WINDOWS.md).**

A reinforcement-learning trading agent that learns entry and exit signals (short/flat/long). It is judged on held-out validation data during training. When it runs a **losing strategy** or breaches a **drawdown limit**, the current generation is **terminated**: its state is wiped and a new generation is spawned. This repeats until a generation reaches the target Sharpe or `max_generations` runs out.

```bash
pip install -r requirements.txt
python main.py --symbol BTC/USDT --timeframe 1h --inherit elite --kill-drawdown 0.35   # Hyperliquid (default)
python main.py --funding --indicators                                                   # optional extra features
python main.py --exchange okx --symbol ETH/USDT                                         # any ccxt exchange, more history
python main.py --synthetic                                                              # offline test data
python main.py --help                                                                   # every knob
```
Each coin and timeframe gets its own model in `runs/<SYMBOL>_<TIMEFRAME>/`. The default source is **Hyperliquid**: no API key, no region blocks, and perp funding rates, but only the latest 5000 candles (~7 months at 1h). For longer history, use `--exchange okx` or `kraken`. Binance and Bybit block some regions.

| file | role |
|---|---|
| `data.py` | crypto candles via ccxt, CSV loader, synthetic prices, chronological split |
| `env.py` | features (no look-ahead), position simulation with costs, metrics |
| `agent.py` | linear softmax policy trained with policy gradient |
| `evolve.py` | generation loop: train → judge → terminate & wipe → respawn |
| `main.py` | CLI; writes `runs/generations.csv`, `champion.npz`, `result.json` |
| `signals.py` | Telegram alerts: OPEN / CLOSE, LONG / SHORT, per coin; optional paper trades |
| `risk.py` | stop-loss, profit lock, min hold, re-entry cooldown, enforced in tests and live |
| `news.py` | News AI: headlines → Claude → alert / pause entries / close opposing positions |
| `report.py` | live trade stats + News AI scorecard |
| `ai4trade.py` | paper trading on [AI-Trader](https://github.com/HKUDS/AI-Trader) (ai4trade.ai) |

## Telegram trade alerts
You get one message per action:

| Alert | Meaning |
|---|---|
| 🟢 OPEN LONG | BUY to open a long |
| 🔴 OPEN SHORT | SELL to open a short |
| ✅/❌ CLOSE LONG | SELL to close the long (shows entry, exit and % result) |
| ✅/❌ CLOSE SHORT | BUY to close the short |

A reversal sends CLOSE and then OPEN. Signals use **closed candles only**, and the bot checks just after each candle closes.

1. In Telegram, message **@BotFather**, send `/newbot`, and copy the token.
2. Send any message to your bot, then open `https://api.telegram.org/bot<TOKEN>/getUpdates` and copy the `chat` → `id`.
3. Train one model per coin, then start the alerts:
```bash
export TELEGRAM_BOT_TOKEN=...  TELEGRAM_CHAT_ID=...      # never commit these
python main.py --symbol BTC/USDT --timeframe 1h --inherit elite --kill-drawdown 0.35
python main.py --symbol ETH/USDT --timeframe 1h --inherit elite --kill-drawdown 0.35
python signals.py --symbols BTC/USDT,ETH/USDT --timeframe 1h          # runs until stopped
python signals.py --symbols BTC/USDT --once --dry-run                 # test: print, don't send
```
Open positions are tracked in `runs/signal_state.json`, so a restart doesn't repeat alerts.

## Ensemble and position sizing
`--ensemble 5` trains 5 models with different seeds that **vote** (they average their action probabilities). It's the default in `start.sh`/`run.py`. `--sizing true` shrinks each new trade (down to 25%) when the last 24h were wilder than the last 30 days. Sizing is off by default.

Measured on Hyperliquid 1h data: 25 single models vs 5 ensembles of 5, unseen test period, risk rules on.

| Coin | Single model: avg (worst … best) | Ensemble of 5: avg (worst … best) | + sizing (ensemble) | Buy & hold |
|---|---|---|---|---|
| BTC | +10.6% (+5.3 … +14.1) | **+12.0% (+11.3 … +12.4)** | +9.3%, drawdown 6.3% → 4.2% | +6.3% |
| ETH | +27.7% (+10.6 … +36.9) | **+32.2% (+30.6 … +33.7)** | +23.0% | +2.6% |
| SOL | +1.1% (−10.4 … +19.2) | +0.5% (−0.8 … +1.7) | −5.6% | +5.2% |

The ensemble removed most of the luck between runs and raised the average on BTC and ETH. Sizing cut drawdown a little but lowered returns and Sharpe on every coin, so it's off by default.

## Risk rules (`risk.py`)
Adapted from [NoFxAiOS/nofx](https://github.com/NoFxAiOS/nofx): "the model proposes, the runtime disposes". The model suggests a position, and these rules decide what is actually held. The rules are identical in backtests and in live alerts.

| Rule | Default | Flag |
|---|---|---|
| Stop-loss | close at −3% | `--stop-loss 0.03` |
| Profit lock | after +5% peak, close if 40% of the gain is given back | `--protect-after 0.05 --giveback 0.4` |
| Minimum hold | 2 candles before the model may exit or flip | `--min-hold-bars 2` |
| Re-entry cooldown | 4 candles flat after any close | `--cooldown-bars 4` |

Disable them with `--use-risk false`. OPEN alerts show the stop-loss price; CLOSE alerts give the reason.

**Measured effect** (BTC/USDT 1h, 5 seeds each, unseen test period, after costs):

| Data | Without rules | With rules | Buy & hold |
|---|---|---|---|
| Hyperliquid (~6 weeks of test data) | +3.6% (4/5 seeds profitable) | **+10.9% (5/5)** | +6.3% |
| OKX (~11 weeks of test data) | −14.8% (0/5) | −6.4% (0/5) | +29.7% |

The rules helped on both datasets, but the bot still lost money on OKX during a strong uptrend. The indicator features (`--indicators`) did not improve results, and neither did funding (`--funding`), so both are off by default.

## News AI (`news.py`)
Every `NEWS_INTERVAL_MIN` minutes (default 10), the bot reads fresh headlines from CoinDesk, Cointelegraph, Decrypt and Google News (crypto + macro: Fed, war, sanctions, tariffs, inflation). It asks Claude to rate them and choose **one bounded action**:

| Action | What happens |
|---|---|
| none | nothing (most news) |
| alert | 📰 Telegram message with the summary |
| pause entries | ⏸️ no new positions in the named coins for up to 24h; open ones keep their stops |
| close positions | 🚨 closes positions the news goes **against** (long + bearish, short + bullish) |

News never opens trades. Every judged headline is logged to `runs/news_log.csv` with prices, and `python report.py` scores Claude's bullish/bearish calls 4h and 24h later. Let news do more only if that scorecard proves itself over 50+ calls.

Enable it by setting `ANTHROPIC_API_KEY` in `.env`. Model: `NEWS_MODEL` (default `claude-opus-5-5`; `claude-haiku-5-5` costs far less). Requests use low effort and the API's automatic refusal fallback. If the news check fails, price signals keep running.

## Paper trading on AI-Trader
Each alert can also be placed as a simulated trade on [ai4trade.ai](https://ai4trade.ai) (from [HKUDS/AI-Trader](https://github.com/HKUDS/AI-Trader)): `buy`/`sell` for longs and `short`/`cover` for shorts, at the platform's live price, with $100K of paper money. The result is an independent record of how the signals perform. **Trades there are public** and other agents can copy them.
```bash
python ai4trade.py register --name MyCryptoBot --email you@example.com   # asks for a new password
export AI4TRADE_TOKEN=...                                                # printed by register
python signals.py --symbols BTC/USDT --paper-usd 10000                   # alerts + paper trades
python ai4trade.py status                                                # paper positions
```
If a paper trade fails, the bot sends a ⚠️ Telegram message and keeps running.

## Kill rules (`evolve.Config`)
- `kill_drawdown`: terminate if the validation max drawdown is above this
- `kill_on_loss`: terminate if the validation net return is below zero (`--kill-on-loss false` to disable)
- `grace_episodes`: no kills before this many episodes, because an untrained agent always loses
- `inherit`: `scratch` = full wipe (the original design); `elite` = new generation starts from a mutated copy of the best survivor

## Read before trusting results
- **TEST is the only honest number.** The champion is selected on validation data from many generations, so its validation score is inflated by luck. `main.py` reports results on a test slice that no generation ever saw.
- Costs default to 5 bps per unit of position change. Set `--cost` to match your exchange's fees and slippage.
- Paper-trade before going live.
