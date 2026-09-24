"""The same fail-open AI review bridge used by the M15 live bot."""
from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import os

import requests

log = logging.getLogger("slp2_ai_review")
AI_REVIEW_URL = os.getenv("AI_REVIEW_URL", "http://localhost:8000/api/v1/trade-signal")
AI_REVIEW_API_KEY = os.getenv("AI_REVIEW_API_KEY", "")
AI_REVIEW_TIMEOUT_SECONDS = float(os.getenv("AI_REVIEW_TIMEOUT_SECONDS", "260"))
AI_REVIEW_ENABLED = os.getenv("AI_REVIEW_ENABLED", "true").lower() in ("1", "true", "yes")


# The service answers REJECT with this wording when its OWN inputs failed (market data,
# news/sentiment data, an agent or the LLM) -- an infrastructure failure, not a view on the
# trade. Treated like an unreachable service: fail open (2026-09-24, after OANDA hiccups
# silently cost two real signals).
INFRA_FAILURE_MARKERS = ("failing closed",)


@dataclass
class ReviewResult:
    approved: bool
    decision: str = "APPROVE"
    reason: str = ""
    signal_id: str | None = None


def review_signal(*, symbol: str, side: str, entry: float, stop_loss: float,
                  take_profit: float, volume: float, timeframe: str,
                  strategy: str, equity: float, recent_loss_streak: int = 0,
                  market_open: bool = True) -> ReviewResult:
    """Ask the production AI service. Only an explicit safety/REJECT blocks.

    market_open must reflect the caller's own, actually-observed state --
    it is not derived here. Callers that already required a fresh live quote
    to reach this point (e.g. SLP2.place_order) may safely pass True.
    """
    if not AI_REVIEW_ENABLED:
        return ReviewResult(True, reason="AI review disabled; passthrough")
    payload = {
        "symbol": symbol, "side": "BUY" if side == "long" else "SELL",
        "entry": entry, "stop_loss": stop_loss, "take_profit": take_profit,
        "volume": volume, "timeframe": timeframe, "strategy": strategy,
        "account": {"balance": equity, "market_open": market_open,
                    "recent_loss_streak": recent_loss_streak},
    }
    headers = {"X-API-Key": AI_REVIEW_API_KEY} if AI_REVIEW_API_KEY else {}
    try:
        response = requests.post(AI_REVIEW_URL, json=payload, headers=headers,
                                 timeout=AI_REVIEW_TIMEOUT_SECONDS)
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError) as exc:
        log.warning("AI review unavailable; passing through: %s", exc)
        return ReviewResult(True, reason=f"review unavailable: {exc}")
    decision = str(data.get("decision", "REJECT")).upper()
    reason = str(data.get("reason", ""))
    if decision == "REJECT" and any(m in reason.lower() for m in INFRA_FAILURE_MARKERS):
        log.warning("AI service could not evaluate the signal; passing through: %s", reason[:300])
        return ReviewResult(True, "UNAVAILABLE", f"review unavailable: {reason}", data.get("signal_id"))
    result = ReviewResult(True, decision, reason, data.get("signal_id"))
    if bool(data.get("execution_blocked", False)) and decision != "APPROVE":
        result.approved = False
    elif decision == "REJECT":
        result.approved = False
    elif decision not in ("APPROVE", "MODIFY", "WAIT"):
        result.decision = "APPROVE"
    return result


def log_agent_detail(signal_id: str | None) -> None:
    if not signal_id or not AI_REVIEW_ENABLED:
        return
    url = AI_REVIEW_URL.replace("/api/v1/trade-signal", f"/api/v1/decisions/{signal_id}")
    headers = {"X-API-Key": AI_REVIEW_API_KEY} if AI_REVIEW_API_KEY else {}
    try:
        response = requests.get(url, headers=headers, timeout=AI_REVIEW_TIMEOUT_SECONDS)
        response.raise_for_status()
        for entry in response.json().get("agent_audit_log", []):
            output = json.dumps(entry.get("output") or {}, ensure_ascii=False)
            log.info("[%s] decision=%s | %s", entry.get("agent_name", "?"),
                     entry.get("decision", "?"), output[:2000])
    except (requests.RequestException, ValueError) as exc:
        log.warning("Could not fetch AI detail for %s: %s", signal_id, exc)


def report_outcome(*, signal_id: str | None, profit: float, exit_reason: str,
                   ticket: int | str | None = None, entry_price: float | None = None,
                   close_price: float | None = None, volume: float | None = None) -> None:
    if not signal_id or not AI_REVIEW_ENABLED:
        return
    url = AI_REVIEW_URL.replace("/api/v1/trade-signal", f"/api/v1/decisions/{signal_id}/outcome")
    headers = {"X-API-Key": AI_REVIEW_API_KEY} if AI_REVIEW_API_KEY else {}
    payload = {"profit": profit, "exit_reason": exit_reason, "ticket": str(ticket) if ticket else None,
               "entry_price": entry_price, "close_price": close_price, "volume": volume}
    try:
        requests.post(url, json=payload, headers=headers, timeout=AI_REVIEW_TIMEOUT_SECONDS).raise_for_status()
    except requests.RequestException as exc:
        log.warning("Could not report AI outcome: %s", exc)
