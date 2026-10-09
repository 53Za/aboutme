# Run the bot 24/7 on a server

The bot needs very little: 1 CPU, 1 GB RAM, about 1 GB of disk. A ~$5/month Linux VPS (Ubuntu 22.04 or newer) from any provider is enough. Hyperliquid doesn't block regions, so any server location works.

## 1. Install Docker (once)
```bash
ssh root@YOUR_SERVER_IP
curl -fsSL https://get.docker.com | sh
```

## 2. Get the bot
```bash
git clone -b add-trading-bot https://github.com/53za/aboutme
cd aboutme/trading_bot
cp .env.example .env
nano .env        # paste TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID; set SYMBOLS, e.g. BTC/USDT,ETH/USDT
chmod 600 .env
```

## 3. Start it
```bash
docker compose up -d --build
docker compose logs -f        # first start trains a model per coin (~1 min each), then waits for candles
```
On Telegram you'll get **▶️ Bot started**, then OPEN/CLOSE alerts, and a **💓 Bot is running** status every 24 hours. If market data fails 3 times in a row, you get a **⚠️** alert; once it's working again, a **✅ recovered** message.

It restarts on its own after crashes and server reboots (`restart: unless-stopped`). Trained models and open positions are kept in `runs/`, so restarts don't lose track of trades.

## Everyday commands
| Task | Command |
|---|---|
| Watch logs | `docker compose logs -f` |
| Stop / start | `docker compose stop` / `docker compose start` |
| Update the code | `git pull && docker compose up -d --build` |
| Retrain a coin (takes effect on the next candle, no restart needed) | `docker compose exec bot python main.py --symbol BTC/USDT --timeframe 1h --inherit elite --kill-drawdown 0.35 --kill-on-loss false --target-sharpe 99 --max-generations 20` |
| Add a coin | add it to `SYMBOLS` in `.env`, then `docker compose up -d` |
| Change timeframe | change `TIMEFRAME` in `.env`, then `docker compose up -d` (it trains new models) |
| Paper trading on ai4trade.ai | set `AI4TRADE_TOKEN` in `.env`, then `docker compose up -d` |

If Docker Hub rate-limits the image pull, build from AWS's mirror of the same image:
`docker compose build --build-arg BASE_IMAGE=public.ecr.aws/docker/library/python:3.12-slim`

## Without Docker (systemd)
```bash
sudo useradd -m trader && sudo -iu trader
git clone -b add-trading-bot https://github.com/53za/aboutme && cd aboutme/trading_bot
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env && nano .env && chmod 600 .env
.venv/bin/python main.py --symbol BTC/USDT --timeframe 1h --inherit elite --kill-drawdown 0.35 --kill-on-loss false --target-sharpe 99
exit
sudo cp /home/trader/aboutme/trading_bot/trading-bot.service /etc/systemd/system/
sudo systemctl enable --now trading-bot
journalctl -u trading-bot -f
```

## Security
- `.env` holds your tokens. Keep it `chmod 600` and never commit it (it's in `.gitignore`).
- The bot only **reads** market data and **sends** Telegram messages. It has no exchange keys and can't place real trades.
