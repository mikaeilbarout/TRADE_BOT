"""The bot-side half of two safety rules that only exist end-to-end.

Both were live bugs found on 2026-09-18:
  * the bot traded through every WAIT, including the ones the service
    issues for an imminent high-impact event or a dead quote;
  * (retired 2026-09-19) the bot used to apply AI-modified levels; it no
    longer applies any -- the AI may approve or refuse, never redraw.

The service-side halves are covered in test_decision_policy.py; these pin
down that the bot actually honours them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_LIVE_BOT = Path(__file__).resolve().parents[2] / "live_bot"
if str(_LIVE_BOT) not in sys.path:
    sys.path.insert(0, str(_LIVE_BOT))

from mt5 import ai_review  # noqa: E402




class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


def _review(monkeypatch, payload: dict):
    monkeypatch.setattr(ai_review, "AI_REVIEW_ENABLED", True)
    monkeypatch.setattr(
        ai_review.requests, "post", lambda *a, **k: _FakeResponse(payload)
    )
    return ai_review.review_signal(
        symbol="XAUUSD",
        side="long",
        entry=4390.0,
        stop_loss=4370.0,
        take_profit=4450.0,
        volume=0.06,
        timeframe="M15",
        strategy="donchian_m15",
        equity=25_000.0,
    )


def test_wait_with_execution_blocked_does_not_trade(monkeypatch):
    """The blackout / stale-quote kind of WAIT."""
    result = _review(
        monkeypatch,
        {
            "signal_id": "s1",
            "decision": "WAIT",
            "reason": "high-impact event inside the news blackout window",
            "execution_blocked": True,
            "modified_trade": None,
        },
    )
    assert result.approved is False
    assert result.decision == "WAIT"


def test_wait_with_execution_blocked_ignores_attached_levels(monkeypatch):
    """A blocked WAIT must not trade even when it carries modified levels."""
    result = _review(
        monkeypatch,
        {
            "signal_id": "s2",
            "decision": "WAIT",
            "reason": "market data went stale",
            "execution_blocked": True,
            "modified_trade": {"entry": 4390.0, "stop_loss": 4375.0, "take_profit": 4435.0},
        },
    )
    assert result.approved is False


def test_uncertainty_wait_still_trades(monkeypatch):
    """The evidence-thin kind: deliberately traded through, per the user's
    policy that only REJECT should stop a signal."""
    result = _review(
        monkeypatch,
        {
            "signal_id": "s3",
            "decision": "WAIT",
            "reason": "degraded analysis inputs",
            "execution_blocked": False,
            "modified_trade": None,
        },
    )
    assert result.approved is True
    assert result.decision == "WAIT"


def test_missing_execution_blocked_field_keeps_old_behaviour(monkeypatch):
    """An older service that doesn't send the field must not start
    blocking everything -- this bridge fails open by design."""
    result = _review(
        monkeypatch,
        {"signal_id": "s4", "decision": "WAIT", "reason": "no field", "modified_trade": None},
    )
    assert result.approved is True


def test_reject_still_blocks_without_the_field(monkeypatch):
    result = _review(
        monkeypatch,
        {"signal_id": "s5", "decision": "REJECT", "reason": "bad setup"},
    )
    assert result.approved is False


def test_modify_from_an_older_service_trades_the_original_signal(monkeypatch):
    """MODIFY was retired 2026-09-19. If an older service still sends one,
    its levels are ignored and the bot's own signal is traded -- the bridge
    exposes no modified levels at all any more."""
    result = _review(
        monkeypatch,
        {
            "signal_id": "s6",
            "decision": "MODIFY",
            "reason": "old service",
            "modified_trade": {"entry": 4391.0, "stop_loss": 4380.0, "take_profit": 4420.0},
        },
    )
    assert result.approved is True
    assert result.decision == "APPROVE"
    assert not hasattr(result, "modified_stop")

