from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.models.trade import ModifiedTrade
from app.services.risk_service import AccountState, RiskService, monetary_risk
from tests.conftest import make_account, make_market_snapshot, make_settings, make_signal


def _risk_service(**overrides) -> RiskService:
    return RiskService(make_settings(**overrides))


def test_pre_check_passes_for_healthy_signal():
    result = _risk_service().pre_check(make_signal(), make_account())
    assert result.passed, result.violations


def test_pre_check_fails_on_low_risk_reward():
    rs = _risk_service(min_risk_reward_ratio=3.0)
    result = rs.pre_check(make_signal(), make_account())  # RR = 2.0
    assert not result.passed
    assert any("risk/reward" in v for v in result.violations)


def test_pre_check_fails_when_daily_loss_limit_hit():
    rs = _risk_service(max_daily_loss_pct=2.0)
    result = rs.pre_check(make_signal(), make_account(daily_loss_pct=2.5))
    assert not result.passed
    assert any("daily loss" in v for v in result.violations)


def test_pre_check_fails_when_market_closed():
    result = _risk_service().pre_check(make_signal(), make_account(market_open=False))
    assert not result.passed
    assert any("market is closed" in v for v in result.violations)


def test_pre_check_fails_on_stale_signal():
    rs = _risk_service(max_signal_age_seconds=1.0)
    signal = make_signal(timestamp=datetime.now(timezone.utc) - timedelta(seconds=30))
    result = rs.pre_check(signal, make_account())
    assert not result.passed
    assert any("stale" in v for v in result.violations)


def test_monetary_risk_uses_asset_contract_size():
    # XAUUSD contract size is 100 oz: 10 points x 0.1 lot x 100 = $100.
    assert monetary_risk("XAUUSD", 3650, 3640, 0.1) == 100.0
    # FX default is a 100k standard lot: 0.0010 x 0.1 x 100_000 = $10.
    assert round(monetary_risk("EURUSD", 1.0900, 1.0890, 0.1), 6) == 10.0


def test_risk_per_trade_limit_is_enforced():
    rs = _risk_service(max_risk_per_trade_pct=0.05)  # $50 on a 100k balance
    result = rs.pre_check(make_signal(), make_account())  # risks $100
    assert not result.passed
    assert any("risk per trade" in v for v in result.violations)


def test_risk_per_trade_passes_within_limit():
    rs = _risk_service(max_risk_per_trade_pct=0.5)  # $500 allowed
    result = rs.pre_check(make_signal(), make_account())  # risks $100
    assert result.passed, result.violations


def test_missing_balance_fails_closed_by_default():
    result = _risk_service().pre_check(make_signal(), AccountState(balance=None))
    assert not result.passed
    assert any("account balance not reported" in v for v in result.violations)


def test_missing_balance_allowed_when_explicitly_configured():
    rs = _risk_service(require_account_balance=False)
    result = rs.pre_check(make_signal(), AccountState(balance=None))
    assert result.passed, result.violations


def test_volatility_gate_rejects_stop_inside_the_noise_band():
    rs = _risk_service(max_volatility_atr_multiple=2.0)
    market = make_market_snapshot(atr=50.0)  # ATR 50 vs a 10-point stop
    result = rs.pre_check(make_signal(), make_account(), market=market)
    assert not result.passed
    assert any("volatility too high" in v for v in result.violations)


def test_stale_market_data_is_rejected():
    market = make_market_snapshot(is_stale=True, freshness_seconds=120.0)
    result = _risk_service().pre_check(make_signal(), make_account(), market=market)
    assert not result.passed
    assert any("stale" in v for v in result.violations)


def test_final_guard_rejects_slippage_on_market_order():
    rs = _risk_service(max_slippage_pct=0.05)
    # Live price has moved ~0.27% away from the intended 3640 entry.
    market = make_market_snapshot(last_price=3650.0)
    trade = ModifiedTrade(
        symbol="XAUUSD", side="BUY", entry=3640, stop_loss=3630, take_profit=3670,
        order_type="MARKET",
    )
    result = rs.final_guard(trade, make_account(), market=market)
    assert not result.passed
    assert any("away from intended entry" in v for v in result.violations)


def test_original_signal_is_slippage_checked_as_a_market_order():
    rs = _risk_service(max_slippage_pct=0.05)
    market = make_market_snapshot(last_price=3660.0)  # drifted from the 3650 entry
    result = rs.final_guard(make_signal(), make_account(), market=market)
    assert not result.passed
    assert any("away from intended entry" in v for v in result.violations)


def test_pending_pullback_entry_is_not_treated_as_slippage():
    """The spec's own modification example (3650 -> 3646 to await a
    retracement) must survive the guard: a resting limit order is not a
    slipped market fill."""
    rs = _risk_service(max_slippage_pct=0.05, max_pending_entry_distance_pct=1.0)
    market = make_market_snapshot(last_price=3650.0)
    trade = ModifiedTrade(
        symbol="XAUUSD", side="BUY", entry=3646, stop_loss=3636, take_profit=3670,
        volume=0.1, order_type="LIMIT",
    )
    result = rs.final_guard(
        trade, make_account(), market=market, original_signal=make_signal()
    )
    assert result.passed, result.violations


def test_pending_entry_too_far_from_market_is_rejected():
    rs = _risk_service(max_pending_entry_distance_pct=0.1)
    market = make_market_snapshot(last_price=3650.0)
    trade = ModifiedTrade(
        symbol="XAUUSD", side="BUY", entry=3600, stop_loss=3590, take_profit=3640,
        order_type="LIMIT",
    )
    result = rs.final_guard(trade, make_account(), market=market)
    assert not result.passed
    assert any("pending entry" in v for v in result.violations)


def test_final_guard_rejects_side_flip_against_original_signal():
    signal = make_signal(side="BUY")
    flipped = ModifiedTrade(
        symbol="XAUUSD", side="SELL", entry=3650, stop_loss=3660, take_profit=3620
    )
    result = _risk_service().final_guard(
        flipped, make_account(), market=None, original_signal=signal
    )
    assert not result.passed
    assert any("does not match signal" in v for v in result.violations)


def test_final_guard_rejects_symbol_swap():
    signal = make_signal(symbol="XAUUSD")
    other = ModifiedTrade(
        symbol="BTCUSD", side="BUY", entry=63000, stop_loss=62000, take_profit=65000
    )
    result = _risk_service().final_guard(
        other, make_account(), market=None, original_signal=signal
    )
    assert not result.passed
    assert any("does not match signal" in v for v in result.violations)


def test_final_guard_passes_good_trade():
    trade = ModifiedTrade(
        symbol="XAUUSD", side="BUY", entry=3650, stop_loss=3640, take_profit=3670,
        volume=0.1,
    )
    result = _risk_service().final_guard(
        trade, make_account(), market=make_market_snapshot(), original_signal=make_signal()
    )
    assert result.passed, result.violations
