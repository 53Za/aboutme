# Fail-fast evolutionary trading bot

A reinforcement-learning trading agent that learns entry and exit signals (short/flat/long). It is judged on held-out validation data during training. When it runs a **losing strategy** or breaches a **drawdown limit**, the current generation is **terminated**: its state is wiped and a new generation is spawned. This repeats until a generation reaches the target Sharpe or `max_generations` runs out.

```bash
pip install -r requirements.txt
python main.py --symbol BTC/USDT --timeframe 1h --inherit elite --kill-drawdown 0.35   # OKX history
python main.py --exchange kraken --symbol ETH/USDT                                      # other coin/exchange
python main.py --synthetic                                                              # offline test data
python main.py --help                                                                   # every knob
```
Each coin and timeframe gets its own model in `runs/<SYMBOL>_<TIMEFRAME>/`. Some exchanges block certain countries; Binance and Bybit return 451/403 in some regions, while OKX and Kraken are more widely reachable.

| file | role |
|---|---|
| `data.py` | crypto candles via ccxt, CSV loader, synthetic prices, chronological split |
| `env.py` | features (no look-ahead), position simulation with costs, metrics |
| `agent.py` | linear softmax policy trained with policy gradient |
| `evolve.py` | generation loop: train → judge → terminate & wipe → respawn |
| `main.py` | CLI; writes `runs/generations.csv`, `champion.npz`, `result.json` |
| `signals.py` | Telegram alerts: OPEN / CLOSE, LONG / SHORT, per coin |

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

## Kill rules (`evolve.Config`)
- `kill_drawdown`: terminate if the validation max drawdown is above this
- `kill_on_loss`: terminate if the validation net return is below zero
- `grace_episodes`: no kills before this many episodes, because an untrained agent always loses
- `inherit`: `scratch` = full wipe (the original design); `elite` = new generation starts from a mutated copy of the best survivor

## Read before trusting results
- **TEST is the only honest number.** The champion is selected on validation data from many generations, so its validation score is inflated by luck. `main.py` reports results on a test slice that no generation ever saw.
- Costs default to 5 bps per unit of position change. Set `--cost` to match your exchange's fees and slippage.
- Paper-trade before going live.
