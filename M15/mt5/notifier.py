"""
Telegram notifications for trade open/close events.

Setup:
1. Message @BotFather on Telegram, run /newbot, and get your bot token.
2. Message your new bot once (anything), then visit:
   https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates
   and find your numeric "chat_id" in the JSON response.
3. Put both in your .env file:
   TELEGRAM_BOT_TOKEN=123456:ABC-your-token
   TELEGRAM_CHAT_ID=123456789

If these are not set, notifications are silently skipped (bot keeps trading normally).
"""

import logging
import requests

log = logging.getLogger("telegram_notifier")

# Distinguishes this project's messages from scalp-bot's in a shared Telegram
# chat/bot -- scalp_sample is the walk-forward-validated sister project
# running in parallel for comparison (see mt5/profiles/profile_m1.py).
PROJECT_TAG = "🧪 SAMPLE"


def _send(token: str, chat_id: str, text: str, timeout: int = 10) -> bool:
    if not token or not chat_id:
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        resp = requests.post(
            url,
            data={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
            timeout=timeout,
        )
        if resp.status_code != 200:
            log.warning(f"Telegram notification failed ({resp.status_code}): {resp.text}")
            return False
        return True
    except Exception as e:
        log.warning(f"Telegram notification error: {e}")
        return False


def notify_trade_opened(token, chat_id, *, profile: str, symbol: str, side: str,
                         volume: float, entry_price: float, stop_price: float,
                         target_price: float, risk_amount: float, equity: float):
    side_label = "BUY (long)" if side == "long" else "SELL (short)"
    text = (
        f"{PROJECT_TAG} | 🟢 <b>Trade opened</b> [{profile.upper()}]\n"
        f"Symbol: {symbol}\n"
        f"Side: {side_label}\n"
        f"Volume: {volume:.2f} lots\n"
        f"Entry: {entry_price:.2f}\n"
        f"Stop Loss: {stop_price:.2f}\n"
        f"Take Profit: {target_price:.2f}\n"
        f"Risk amount: {risk_amount:.2f}\n"
        f"Account equity: {equity:.2f}"
    )
    _send(token, chat_id, text)


def notify_trade_closed(token, chat_id, *, profile: str, symbol: str, side: str,
                         volume: float, entry_price: float, close_price: float,
                         profit: float, reason: str, equity: float):
    side_label = "BUY (long)" if side == "long" else "SELL (short)"
    outcome_emoji = "✅" if profit > 0 else "❌" if profit < 0 else "➖"
    text = (
        f"{PROJECT_TAG} | {outcome_emoji} <b>Trade closed</b> [{profile.upper()}]\n"
        f"Symbol: {symbol}\n"
        f"Side: {side_label}\n"
        f"Volume: {volume:.2f} lots\n"
        f"Entry: {entry_price:.2f}\n"
        f"Close: {close_price:.2f}\n"
        f"Reason: {reason}\n"
        f"Profit/Loss: {profit:.2f}\n"
        f"Account equity: {equity:.2f}"
    )
    _send(token, chat_id, text)


def notify_error(token, chat_id, *, profile: str, message: str):
    text = f"{PROJECT_TAG} | ⚠️ <b>Bot error</b> [{profile.upper()}]\n{message}"
    _send(token, chat_id, text)
