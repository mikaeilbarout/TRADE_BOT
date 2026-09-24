# Setting up the XAUUSD bot on Pepperstone (MetaTrader5 + Windows VPS or PC)

## Before anything else
- This account must be a **demo** account. The script prints the account type
  on connect — stop immediately if it doesn't say demo.
- All 5 strategy profiles (M1/M5/M15/M30/H1) were backtested and walk-forward
  validated on historical data — see the main README for each profile's
  Calmar/drawdown — but none of that is a guaranteed money-maker. Watch how
  they perform on new (forward) data before considering real money.
- XAUUSD spreads and slippage can spike during news events (e.g. NFP) — the bot
  has no news filter, so that risk remains.

## 1. Machine requirements
Either your own always-on Windows PC, or a Windows VPS (Vultr, Contabo,
ForexVPS, or Pepperstone's own recommended VPS). 2GB RAM is enough for a VPS.

## 2. Install MT5 and log into the demo account
1. Download and install MetaTrader 5 (from Pepperstone's site or metatrader5.com)
2. Log in with your demo account details (Login, Password, Server — from Pepperstone)
3. Enable the **AutoTrading** button at the top of the terminal (must be green)
4. Make sure **XAUUSD** is visible in Market Watch (right-click -> Show All if not)

## 3. Install Python and packages
```powershell
# install Python from python.org (check "Add to PATH" during setup)
pip install MetaTrader5 pandas numpy python-dotenv requests
```

## 4. Copy the project and set environment variables
Copy the `scalp_sample` folder to the machine, then create a `.env` file:
```
MT5_LOGIN=12345678
MT5_PASSWORD=your_demo_password
MT5_SERVER=mt5-demo01.pepperstone.com
```
(Copy the exact server name from the MT5 login window — it may look like `PepperstoneUK-Demo`.)

### Optional: Telegram notifications
To get a Telegram message every time a trade opens or closes:
1. On Telegram, message **@BotFather** and run `/newbot`, then copy the token it gives you.
2. Send your new bot any message, then open this URL in a browser (with your token):
   `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates`
   and find your numeric `chat_id` in the response.
3. Add both to `.env`:
```
TELEGRAM_BOT_TOKEN=123456:ABC-your-token
TELEGRAM_CHAT_ID=123456789
```
If left blank, the bot just skips notifications — everything else still works normally.

## 5. Running the bot
Five independent strategies are ready, selected with `--profile`:
```powershell
python mt5\live_bot_mt5.py --profile m1
python mt5\live_bot_mt5.py --profile m5
python mt5\live_bot_mt5.py --profile m15
python mt5\live_bot_mt5.py --profile m30
python mt5\live_bot_mt5.py --profile h1
```
Each has its own magic number (M1=991001, M5=991005, M15=991015,
M30=991030, H1=991060), so they can run **simultaneously** (each in its own
terminal window) on the same account without interfering with each other's
positions.

You should see it connect, report the account as "DEMO", and start checking
for signals every 10 seconds.

## 6. Running 24/7 (personal PC or VPS)

### Simple option: watchdog script (recommended for a personal PC)
Run `mt5/run_with_watchdog.bat` with a profile name:
```powershell
mt5\run_with_watchdog.bat m1
mt5\run_with_watchdog.bat m5
mt5\run_with_watchdog.bat m15
mt5\run_with_watchdog.bat m30
mt5\run_with_watchdog.bat h1
```
Run each in its own terminal window so all five run at once. This script:
- Runs the bot with that profile
- Automatically restarts it 10 seconds after any crash

To auto-start on Windows login:
1. Win+R -> `shell:startup` -> Enter
2. Add a shortcut to `run_with_watchdog.bat` in that folder

### More robust option (VPS): NSSM service
```powershell
nssm install ScalpSampleXAUUSD "C:\Python3XX\python.exe" "C:\path\to\scalp_sample\mt5\live_bot_mt5.py --profile m1"
nssm start ScalpSampleXAUUSD
```
This way the bot survives VPS reboots and RDP disconnects, since Windows restarts the service itself. Repeat for m5/m15/m30/h1 with different service names.

### About MT5-connection drops (not the whole system)
The script checks the MT5 terminal connection every 10 seconds; if it drops
(e.g. a brief internet blip), it retries with increasing backoff (5, 10, 20...
up to 5 minutes) automatically — no manual action needed. This is separate
from a full process restart (handled by the watchdog/NSSM); the two work together.

## 7. Monitoring over the following days
- Logs are saved to `logs/mt5_bot_<profile>_YYYYMMDD.log` — check once a day
- Check balance and equity in MT5 daily
- If a strategy hits its daily loss cap (3%), it pauses itself until the next day — no need to stop it manually
- If Telegram is set up, you'll get a message on every trade open/close automatically

## After the test period
Gather the results (number of trades, win rate, P&L) per profile and share
them so we can review whether parameters need adjusting — before considering
a live account.
