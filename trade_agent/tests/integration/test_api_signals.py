from __future__ import annotations

from fastapi.testclient import TestClient

from app.bootstrap import build_container
from app.main import create_app
from app.models.enums import ApprovalStatus, FinalDecision, TradingMode
from tests.conftest import TEST_BALANCE, make_llm, make_settings


def _client(**settings_overrides) -> TestClient:
    # Pinned to four_agent regardless of Settings' own default
    # (single_agent since 2026-09-19): these tests assert on the API
    # contract and audit trail via the four-agent chain's own result
    # shape (news_result/sentiment_result/technical_result/final_result),
    # not on either pipeline's decision logic.
    settings = make_settings(
        database_url="sqlite+aiosqlite:///:memory:",
        pipeline_mode="four_agent",
        **settings_overrides,
    )
    container = build_container(settings, llm_override=make_llm())
    return TestClient(create_app(container=container))


def _payload(**overrides) -> dict:
    payload = {
        "symbol": "XAUUSD",
        "side": "BUY",
        "entry": 3650.20,
        "stop_loss": 3640.20,
        "take_profit": 3670.20,
        "volume": 0.10,
        "timeframe": "M5",
        "strategy": "scalp_strategy",
        "account": {"balance": TEST_BALANCE, "market_open": True},
    }
    payload.update(overrides)
    return payload


def test_trade_signal_endpoint_matches_api_contract():
    with _client() as client:
        r = client.post("/api/v1/trade-signal", json=_payload())
        assert r.status_code == 200, r.text
        body = r.json()
        assert {"signal_id", "decision", "confidence", "reason", "modified_trade"} <= set(body)
        assert body["decision"] == FinalDecision.APPROVE.value


def test_audit_trail_stores_each_agent_input_and_output():
    with _client() as client:
        signal_id = client.post("/api/v1/trade-signal", json=_payload()).json()["signal_id"]
        record = client.get(f"/api/v1/decisions/{signal_id}").json()

        assert record["news_result"] and record["sentiment_result"]
        assert record["technical_result"] and record["final_result"]
        assert record["data_sources"]["llm_provider"] == "mock"

        logs = {log["agent_name"]: log for log in record["agent_audit_log"]}
        assert set(logs) == {
            "news_agent",
            "sentiment_agent",
            "technical_agent",
            "final_decision_agent",
        }
        # Each agent's snapshot is ITS OWN input, not a copy of the signal.
        # Parallelized 2026-09-18: technical/sentiment no longer see an
        # upstream_chain (they run concurrently with news, not after it) --
        # only final_decision_agent's snapshot carries the full agent_chain.
        assert "upstream_chain" not in logs["technical_agent"]["input_snapshot"]
        assert "market_conditions" in logs["technical_agent"]["input_snapshot"]
        assert "sentiment_sources" in logs["sentiment_agent"]["input_snapshot"]
        assert [link["agent"] for link in logs["final_decision_agent"]["input_snapshot"]["agent_chain"]] == [
            "news_agent",
            "sentiment_agent",
            "technical_agent",
        ]
        assert logs["news_agent"]["model"]
        assert logs["news_agent"]["model_version"] == "mock"


def test_decision_lookup_404_for_unknown_signal():
    with _client() as client:
        assert client.get("/api/v1/decisions/does-not-exist").status_code == 404


def test_shadow_mode_never_returns_actionable_decision():
    with _client(trading_mode=TradingMode.SHADOW) as client:
        body = client.post("/api/v1/trade-signal", json=_payload()).json()
        assert body["decision"] == "SHADOW_OBSERVE_ONLY"
        assert body["modified_trade"] is None
        assert body["ai_decision"] == FinalDecision.APPROVE.value


def test_paper_mode_records_hypothetical_execution():
    with _client(trading_mode=TradingMode.PAPER) as client:
        body = client.post("/api/v1/trade-signal", json=_payload()).json()
        assert body["paper_trade"] is True
        fill = body["hypothetical_execution"]
        assert fill["filled"] is True
        assert fill["fill_price"] > 0
        assert "slippage_pct" in fill


def test_manual_mode_parks_decision_for_human_review():
    with _client(trading_mode=TradingMode.MANUAL) as client:
        body = client.post("/api/v1/trade-signal", json=_payload()).json()
        assert body["requires_human_approval"] is True
        assert body["approval_status"] == ApprovalStatus.AWAITING_HUMAN.value

        pending = client.get("/api/v1/decisions/pending").json()
        assert len(pending) == 1
        # A reviewer sees the whole chain without a second call.
        assert pending[0]["news_result"] and pending[0]["technical_result"]


def test_human_can_approve_a_pending_decision():
    with _client(trading_mode=TradingMode.MANUAL, require_account_balance=False) as client:
        signal_id = client.post("/api/v1/trade-signal", json=_payload()).json()["signal_id"]
        r = client.post(
            f"/api/v1/decisions/{signal_id}/approve",
            json={"reviewer": "mikaeil", "note": "checked the chart"},
        )
        assert r.status_code == 200, r.text
        assert r.json()["approval_status"] == ApprovalStatus.HUMAN_APPROVED.value
        assert r.json()["execute"] is True

        # Second approval is a conflict, not a silent re-approval.
        assert client.post(f"/api/v1/decisions/{signal_id}/approve", json={}).status_code == 409


def test_human_can_reject_a_pending_decision():
    with _client(trading_mode=TradingMode.MANUAL) as client:
        signal_id = client.post("/api/v1/trade-signal", json=_payload()).json()["signal_id"]
        r = client.post(f"/api/v1/decisions/{signal_id}/reject", json={"reviewer": "mikaeil"})
        assert r.status_code == 200
        assert r.json()["execute"] is False
        assert client.get("/api/v1/decisions/pending").json() == []


def test_human_approval_cannot_bypass_hard_risk_rules():
    """A human reviewing later is still subject to the deterministic rules --
    here the balance is unknown at approval time, so the guard blocks it."""
    with _client(trading_mode=TradingMode.MANUAL, require_account_balance=True) as client:
        signal_id = client.post("/api/v1/trade-signal", json=_payload()).json()["signal_id"]
        r = client.post(f"/api/v1/decisions/{signal_id}/approve", json={})
        assert r.status_code == 409
        assert "violations" in r.json()["detail"]


def test_invalid_signal_direction_returns_422():
    with _client() as client:
        r = client.post("/api/v1/trade-signal", json=_payload(stop_loss=3660.0))
        assert r.status_code == 422


def test_api_key_is_enforced_when_configured():
    with _client(internal_api_key="s3cret") as client:
        assert client.post("/api/v1/trade-signal", json=_payload()).status_code == 401
        assert client.get("/api/v1/decisions").status_code == 401

        ok = client.post(
            "/api/v1/trade-signal", json=_payload(), headers={"X-API-Key": "s3cret"}
        )
        assert ok.status_code == 200
        # Health stays open for liveness probes and leaks no secrets.
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["auth_enabled"] is True
        assert "s3cret" not in health.text


def test_api_key_not_required_when_unset():
    with _client() as client:
        assert client.post("/api/v1/trade-signal", json=_payload()).status_code == 200
