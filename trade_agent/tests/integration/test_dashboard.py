"""Dashboard page: public HTML shell, data still behind the API key."""
from __future__ import annotations

from tests.integration.test_api_signals import _client, _payload


def test_dashboard_page_is_served_without_a_key():
    with _client(internal_api_key="s3cret") as client:
        for path in ("/", "/dashboard"):
            resp = client.get(path)
            assert resp.status_code == 200
            assert resp.headers["content-type"].startswith("text/html")
            assert "Gold Trading Dashboard" in resp.text
            assert "s3cret" not in resp.text


def test_dashboard_data_still_requires_the_key():
    with _client(internal_api_key="s3cret") as client:
        assert client.get("/api/v1/decisions").status_code == 401
        assert client.get("/api/v1/decisions", headers={"X-API-Key": "s3cret"}).status_code == 200


def test_decision_list_exposes_strategy_and_reason():
    with _client() as client:
        client.post("/api/v1/trade-signal", json=_payload(strategy="slp2_m15"))
        rows = client.get("/api/v1/decisions").json()
        assert rows[0]["strategy"] == "slp2_m15"
        assert isinstance(rows[0]["reason"], str) and rows[0]["reason"]
