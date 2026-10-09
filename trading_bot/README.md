# Fail-fast evolutionary trading bot

A reinforcement-learning trading agent that learns entry and exit signals (short/flat/long). It is judged on held-out validation data during training. When it runs a **losing strategy** or breaches a **drawdown limit**, the current generation is **terminated**: its state is wiped and a new generation is spawned. This repeats until a generation reaches the target Sharpe or `max_generations` runs out.

```bash
pip install -r requirements.txt
python main.py                                  # synthetic data, full wipe on death
python main.py --csv prices.csv                 # your data (needs a `close` column)
python main.py --inherit elite --kill-drawdown 0.35
python main.py --help                           # every knob in evolve.Config
```

| file | role |
|---|---|
| `data.py` | CSV loader, synthetic regime-switching prices, chronological train/val/test split |
| `env.py` | features (no look-ahead), position simulation with costs, metrics |
| `agent.py` | linear softmax policy trained with policy gradient |
| `evolve.py` | generation loop: train → judge → terminate & wipe → respawn |
| `main.py` | CLI; writes `runs/generations.csv`, `champion.npz`, `result.json` |
| `signals.py` | sends LONG / SHORT / FLAT alerts from the champion to Telegram |

## Telegram signals
1. In Telegram, message **@BotFather**, send `/newbot`, and copy the token it gives you.
2. Send any message to your new bot, then open `https://api.telegram.org/bot<TOKEN>/getUpdates` and copy the `chat` → `id` value.
3. Train a model, then run the notifier:
```bash
export TELEGRAM_BOT_TOKEN=...  TELEGRAM_CHAT_ID=...
python main.py --inherit elite --kill-drawdown 0.35          # writes runs/champion.npz
python signals.py --csv prices.csv --dry-run                  # test: print instead of send
pip install ccxt                                              # for live exchange candles
python signals.py --exchange binance --symbol BTC/USDT --timeframe 1h --every 3600
```
A message is sent only when the signal changes, so you get one alert per entry or exit. Train on candles with the same timeframe that you run live.

## Kill rules (`evolve.Config`)
- `kill_drawdown`: terminate if the validation max drawdown is above this
- `kill_on_loss`: terminate if the validation net return is below zero
- `grace_episodes`: no kills before this many episodes, because an untrained agent always loses
- `inherit`: `scratch` = full wipe (the original design); `elite` = new generation starts from a mutated copy of the best survivor

## Read before trusting results
- **TEST is the only honest number.** The champion is selected on validation data from many generations, so its validation score is inflated by luck. `main.py` reports results on a test slice that no generation ever saw.
- Costs default to 5 bps per unit of position change. Set `--cost` to match your broker's fees and slippage.
- Paper-trade before going live.
