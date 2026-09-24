"""Bridge between the live MT5 bot and the AI decision-review service.

The bot still generates every signal and still owns execution -- this module
only asks the review service what it thinks BEFORE `place_order` is called.
Nothing here talks to MT5 or a broker directly.

The AI layer is a helper that catches the specific trades it is confident
are likely losers -- it is not a gate the bot depends on to function. So the
fail direction is OPEN, not closed: a signal is only ever blocked when the
service was actually reached and either answered REJECT or set
`execution_blocked` on its answer. If the service is disabled, unreachable,
times out, returns something unparseable, or answers with anything else
unexpected, this passes the ORIGINAL signal straight through unmodified --
the underlying strategy already trades profitably on its own (see the
baseline backtest), and an AI outage should not stop it from running.

`execution_blocked` exists because the decision label is not enough: WAIT
means both "not confident" (trade anyway, per the user's policy) and "an
FOMC print lands in three minutes" / "the quote this was decided on is
already dead" (do not trade). The service knows which; the label does not.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass

import requests

log = logging.getLogger("xauusd_bot_ai_review")

AI_REVIEW_URL = os.getenv("AI_REVIEW_URL", "http://localhost:8000/api/v1/trade-signal")
AI_REVIEW_API_KEY = os.getenv("AI_REVIEW_API_KEY", "")  # matches INTERNAL_API_KEY; blank = auth disabled
AI_REVIEW_TIMEOUT_SECONDS = float(os.getenv("AI_REVIEW_TIMEOUT_SECONDS", "260"))
# news/sentiment/technical run CONCURRENTLY, then final decision -- so the
# worst case is 2 sequential agent slots, each with its own
# AGENT_TIMEOUT_SECONDS and one retry (max_attempts=2). This must exceed
# 2 * AGENT_TIMEOUT_SECONDS * 2 plus the market-data and news/sentiment
# fetches, or the bot gives up and fails open on a request the service would
# have answered. Left generous on purpose -- keep in sync with the service's
# own AGENT_TIMEOUT_SECONDS if that ever changes.

AI_REVIEW_ENABLED = os.getenv("AI_REVIEW_ENABLED", "true").lower() in ("1", "true", "yes")


@dataclass
class ReviewResult:
    approved: bool
    decision: str = "APPROVE"
    reason: str = ""
    # The service's own record id for this review, needed later to report
    # what actually happened via report_outcome(). None whenever the service
    # was never actually reached (disabled, unreachable, malformed response)
    # -- there is no server-side record to attach an outcome to in that case.
    signal_id: str | None = None


def _side_word(side: str) -> str:
    """TradePlan uses "long"/"short"; the API uses BUY/SELL."""
    return "BUY" if side == "long" else "SELL"


def review_signal(
    *,
    symbol: str,
    side: str,
    entry: float,
    stop_loss: float,
    take_profit: float,
    volume: float,
    timeframe: str,
    strategy: str,
    equity: float,
    recent_loss_streak: int = 0,
) -> ReviewResult:
    """Ask the AI review service whether to take this trade.

    Fails OPEN: returns approved=True (the original signal, unmodified)
    unless the service was actually reached and explicitly answered REJECT
    or WAIT. The caller should treat approved=False as "the AI specifically
    flagged this one," never as "we couldn't get an opinion."
    """
    if not AI_REVIEW_ENABLED:
        return ReviewResult(approved=True, decision="APPROVE", reason="AI review disabled; passthrough")

    payload = {
        "symbol": symbol,
        "side": _side_word(side),
        "entry": entry,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        # Actual broker lots (contract_size already divided out) -- the
        # service's own risk gate multiplies by contract_size itself
        # (app/services/risk_service.py: monetary_risk = |entry-stop| *
        # volume * contract_size). The caller must convert plan.position_size
        # (risk_amount / stop_distance) to lots the same way place_order()
        # does before passing it here, or the risk check overstates monetary
        # risk by exactly contract_size and rejects every signal.
        "volume": volume,
        "timeframe": timeframe,
        "strategy": strategy,
        "account": {
            "balance": equity,
            "market_open": True,
            "recent_loss_streak": recent_loss_streak,
        },
    }
    headers = {"X-API-Key": AI_REVIEW_API_KEY} if AI_REVIEW_API_KEY else {}

    try:
        resp = requests.post(
            AI_REVIEW_URL, json=payload, headers=headers, timeout=AI_REVIEW_TIMEOUT_SECONDS
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        log.warning(f"AI review service unreachable ({exc}); trading this signal unreviewed.")
        return ReviewResult(approved=True, decision="APPROVE", reason=f"review unavailable: {exc}")
    except ValueError as exc:  # malformed JSON
        log.warning(f"AI review service returned unparseable response ({exc}); trading this signal unreviewed.")
        return ReviewResult(approved=True, decision="APPROVE", reason=f"malformed response: {exc}")

    decision = str(data.get("decision", "REJECT")).upper()
    reason = str(data.get("reason", ""))
    signal_id = data.get("signal_id")

    # The service's explicit "do not trade this" flag, independent of the
    # label. WAIT covers two different things -- "not enough evidence"
    # (which this bot deliberately trades through, see the WAIT branch
    # below) and "an imminent high-impact event / the quote this was
    # decided on is already dead" (which it must NOT trade through). Only
    # the service can tell them apart, so it now says so directly.
    # Defaults to False so an older service that doesn't send the field
    # keeps the previous behavior rather than blocking everything.
    if bool(data.get("execution_blocked", False)) and decision != "APPROVE":
        log.info(f"AI review: {decision} with execution_blocked -- not trading. {reason}")
        return ReviewResult(
            approved=False, decision=decision, reason=reason, signal_id=signal_id
        )

    if decision == "APPROVE":
        return ReviewResult(approved=True, decision=decision, reason=reason, signal_id=signal_id)

    if decision == "MODIFY":
        # Retired 2026-09-19: the AI may approve or refuse a trade, never
        # redraw it. The current service cannot emit MODIFY at all; if an
        # older one does, the attached levels are deliberately IGNORED and
        # the bot's own signal is traded (a MODIFY is, at bottom, "the
        # direction is fine").
        log.info("AI review: MODIFY received -- levels ignored, trading the original signal.")
        return ReviewResult(approved=True, decision="APPROVE", reason=reason, signal_id=signal_id)

    if decision == "REJECT":
        # The only case this module actually blocks a trade: the service was
        # reached and explicitly said this specific setup is a probable loser.
        # signal_id carried through like every other branch, below, so a
        # REJECT can still be looked up via log_agent_detail()/the dashboard.
        return ReviewResult(approved=False, decision=decision, reason=reason, signal_id=signal_id)

    if decision == "WAIT":
        # Reaching here means execution_blocked was False, i.e. the service
        # classified this WAIT as an evidence problem rather than a safety
        # one (the safety kind returns above and does not trade).
        #
        # WAIT means "not enough evidence to be confident either way" (missing
        # news/sentiment, low confidence) -- NOT "this trade is a probable
        # loser." Per the user: the AI's job is narrowly to catch high-
        # confidence losers (REJECT) and raise win rate by cutting those;
        # everything else, including WAIT, should trade normally. Blocking on
        # WAIT would make a cautious/uncertain read behave like a rejection it
        # never actually was. The label is kept (decision="WAIT") for the
        # audit trail -- only the blocking behavior changes.
        #
        return ReviewResult(approved=True, decision=decision, reason=reason, signal_id=signal_id)

    # Anything else (an unrecognized decision string) is a response-parsing
    # surprise, not a considered "no" -- pass the signal through rather than
    # silently blocking trades on a value this bot doesn't understand.
    log.warning(f"AI review returned unexpected decision {decision!r}; trading this signal unreviewed.")
    return ReviewResult(approved=True, decision="APPROVE", reason=f"unexpected decision {decision!r}: {reason}")


def log_agent_detail(signal_id: str | None) -> None:
    """Fetch and log each agent's full input/output for this signal_id, so
    the local bot log carries the whole 4-agent conversation (news,
    sentiment, technical, final) instead of just the one-line final reason
    review_signal() already logs above this call.

    Best-effort and silent on failure, same as report_outcome(): this is a
    convenience for reading the log later, and the review service already
    made its decision by the time this runs, so nothing here can affect
    trading. signal_id is None whenever the service was never actually
    reached (disabled, unreachable, malformed response) -- there is no
    server-side record to fetch in that case.
    """
    if not signal_id or not AI_REVIEW_ENABLED:
        return

    url = AI_REVIEW_URL.replace("/api/v1/trade-signal", f"/api/v1/decisions/{signal_id}")
    headers = {"X-API-Key": AI_REVIEW_API_KEY} if AI_REVIEW_API_KEY else {}
    try:
        resp = requests.get(url, headers=headers, timeout=AI_REVIEW_TIMEOUT_SECONDS)
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError) as exc:
        log.warning(f"could not fetch agent detail for {signal_id} ({exc})")
        return

    for entry in data.get("agent_audit_log", []):
        name = entry.get("agent_name", "?")
        agent_decision = entry.get("decision", "?")
        output = json.dumps(entry.get("output") or {}, ensure_ascii=False)
        if len(output) > 2000:
            output = output[:2000] + "...(truncated)"
        log.info(f"  [{name}] decision={agent_decision} | {output}")


def report_outcome(
    *,
    signal_id: str | None,
    profit: float,
    exit_reason: str,
    ticket: int | str | None = None,
    entry_price: float | None = None,
    close_price: float | None = None,
    volume: float | None = None,
) -> None:
    """Tell the review service what actually happened to a position it
    approved, so decisions can eventually be judged against real outcomes
    (not just simulated ones) rather than staying opinions forever.

    Best-effort and silent on failure -- this runs after the trade is
    already closed in MT5, so nothing here should ever block or retry; a
    missed report just means that one row's outcome stays unrecorded.
    """
    if not signal_id or not AI_REVIEW_ENABLED:
        return

    url = AI_REVIEW_URL.replace("/api/v1/trade-signal", f"/api/v1/decisions/{signal_id}/outcome")
    payload = {
        "profit": profit,
        "exit_reason": exit_reason,
        "ticket": str(ticket) if ticket is not None else None,
        "entry_price": entry_price,
        "close_price": close_price,
        "volume": volume,
    }
    headers = {"X-API-Key": AI_REVIEW_API_KEY} if AI_REVIEW_API_KEY else {}
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=AI_REVIEW_TIMEOUT_SECONDS)
        resp.raise_for_status()
    except requests.RequestException as exc:
        log.warning(f"could not report trade outcome for {signal_id} ({exc}); dashboard will miss this one.")
