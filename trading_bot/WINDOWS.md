# Run the bot on your Windows PC

The bot runs as long as the PC is **on, awake and online**. If it sleeps or shuts down, alerts stop until you start it again. For true 24/7, use a server ([DEPLOY.md](DEPLOY.md)).

## 1. Install Python (once)
1. Go to https://www.python.org/downloads/ and click **Download Python**.
2. Run the installer and **tick "Add python.exe to PATH"** at the bottom of the first screen, then click **Install Now**.

## 2. Download the bot
1. Open https://github.com/53za/aboutme/archive/refs/heads/add-trading-bot.zip (the download starts).
2. Right-click the ZIP → **Extract All…** → choose e.g. `C:\bot`.
3. Open `C:\bot\aboutme-add-trading-bot\trading_bot`.

## 3. Start it
1. Double-click **`start_windows.bat`**. The first run installs what it needs, which takes a few minutes.
2. Notepad opens the settings file `.env`. Fill in:
   ```
   TELEGRAM_BOT_TOKEN=your-new-token-from-BotFather
   TELEGRAM_CHAT_ID=1019173226
   ```
   Save (**Ctrl+S**) and close Notepad.
3. Double-click **`start_windows.bat`** again. It trains BTC, ETH and SOL models (about 1 minute each), then you get **▶️ Bot started** on Telegram.

**Leave the black window open**; closing it stops the bot. If the bot crashes, the window restarts it after 30 seconds.

If Windows shows **"Windows protected your PC"**, click **More info → Run anyway**. This appears for any downloaded `.bat` file.

## 4. Keep it running
- **Stop the PC from sleeping:** Settings → System → Power → **Screen and sleep** → "When plugged in, put my device to sleep after" → **Never**. The screen can still turn off.
- **Start automatically when you log in:** press **Win+R**, type `shell:startup`, press Enter. Then right-click `start_windows.bat` → **Show more options → Create shortcut**, and move the shortcut into the folder that opened.

## Settings (`.env`, edit with Notepad)
| Setting | Example | Meaning |
|---|---|---|
| `SYMBOLS` | `BTC/USDT,ETH/USDT,SOL/USDT` | coins to watch |
| `TIMEFRAME` | `1h` | candle size (changing it trains new models) |
| `SIGNAL_ARGS` | `--heartbeat-hours 12` | extra options |
| `AI4TRADE_TOKEN` | | optional paper trading on ai4trade.ai |

After you change `.env`, close the black window and double-click `start_windows.bat` again.
