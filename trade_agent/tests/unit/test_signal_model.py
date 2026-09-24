from __future__ import annotations

import pytest
from pydantic import ValidationError

from tests.conftest import make_signal


def test_buy_requires_sl_below_entry_below_tp():
    with pytest.raises(ValidationError):
        make_signal(side="BUY", entry=100, stop_loss=110, take_profit=120)


def test_sell_requires_tp_below_entry_below_sl():
    with pytest.raises(ValidationError):
        make_signal(side="SELL", entry=100, stop_loss=90, take_profit=80)


def test_valid_sell_signal():
    s = make_signal(side="SELL", entry=100, stop_loss=105, take_profit=90)
    assert s.risk_reward_ratio == pytest.approx(2.0)


def test_symbol_normalized_to_uppercase():
    s = make_signal(symbol="xauusd")
    assert s.symbol == "XAUUSD"


def test_risk_reward_ratio_computation():
    s = make_signal(entry=100, stop_loss=95, take_profit=110)
    assert s.risk_reward_ratio == pytest.approx(2.0)
