"""Best-effort Telegram notifications for SLP2."""
import logging
import os
import requests

log = logging.getLogger("slp2_telegram")
TAG = "🤖 SLP2"


def _send(text: str) -> None:
    token, chat_id = os.getenv("TELEGRAM_BOT_TOKEN", ""), os.getenv("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        return
    try:
        response = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                                 data={"chat_id": chat_id, "text": text, "parse_mode": "HTML"}, timeout=10)
        response.raise_for_status()
    except requests.RequestException as exc:
        log.warning("Telegram notification failed: %s", exc)


def opened(*, side, volume, entry, stop, target, equity):
    _send(f"{TAG} | 🟢 <b>Trade opened</b> [M15]\n"
          f"XAUUSD {side.upper()} | {volume:.2f} lots\n"
          f"Entry: {entry:.2f} | SL: {stop:.2f} | TP: {target:.2f}\n"
          f"Equity: {equity:.2f}")


def closed(*, side, volume, entry, close, profit, reason, equity):
    emoji = "✅" if profit > 0 else "❌" if profit < 0 else "➖"
    _send(f"{TAG} | {emoji} <b>Trade closed</b> [M15]\n"
          f"XAUUSD {side.upper()} | {volume:.2f} lots\n"
          f"Entry: {entry:.2f} | Close: {close:.2f}\n"
          f"{reason} | P/L: {profit:.2f} | Equity: {equity:.2f}")


def error(message: str):
    _send(f"{TAG} | ⚠️ <b>Bot error</b> [M15]\n{message}")
