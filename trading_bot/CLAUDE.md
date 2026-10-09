# Trading bot — notes for Claude Code running on the live server

This folder runs a **live** crypto signal bot in Docker (`docker compose`, service `bot`). It sends OPEN/CLOSE LONG/SHORT alerts to the owner's Telegram for BTC, ETH and SOL. It has **no exchange keys** and never places real trades.

## Look, don't break
- Status: `docker compose ps` · logs: `docker compose logs --tail 100`
- Open positions: `runs/signal_state.json` · closed trades: `runs/trades.csv`
- News AI: decisions in `runs/news_log.csv`, active pauses in `runs/news_state.json`; scorecard in `report.py`
- Performance: `docker compose exec bot python report.py`
- Files in `runs/` are written by the running bot. Read them, never edit them.

## Rules
1. **Never print, copy, commit or send the contents of `.env`.** It holds the Telegram, Anthropic and AI-Trader keys.
2. **Ask the owner before** restarting or stopping the bot, retraining a live model, changing `.env`, or deploying code changes.
3. **Prove an improvement before deploying it.** Compare against the current version over 5+ seeds on the unseen test split (`main.py` prints TEST), and against buy-and-hold. Validation scores alone are inflated by selection and don't count. Never pick a model or setting by its test score.
4. Back up a model before replacing it: `cp -r runs/BTC_USDT_1h runs/BTC_USDT_1h.bak-$(date +%F)`.
5. Treat live results as the real test. With fewer than ~30 closed trades, conclusions are noise, so say so.

## Deploying a change (after the owner agrees)
```bash
git pull                                 # or edit + test locally first
docker compose up -d --build             # restarts the bot; positions in runs/ are kept
docker compose logs -f --tail 20         # confirm "Bot started"; Ctrl+C to stop watching
```
Retrained models are picked up on the next candle without a restart:
`docker compose exec bot python main.py --symbol BTC/USDT --timeframe 1h --inherit elite --kill-drawdown 0.35 --kill-on-loss false --target-sharpe 99 --max-generations 20`

## Code map
`main.py` train · `evolve.py` generations + kill rules · `agent.py` policy · `env.py` features/backtest · `risk.py` stop-loss etc. · `signals.py` live loop + Telegram · `report.py` live stats + news scorecard · `news.py` News AI · `ai4trade.py` paper trading
