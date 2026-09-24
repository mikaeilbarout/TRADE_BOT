"""
Risk management: position sizing, stop-loss / take-profit calculation.
"""

from dataclasses import dataclass


@dataclass
class TradePlan:
    side: str            # "long" or "short"
    entry_price: float
    stop_price: float
    target_price: float
    position_size: float  # in base-asset units (e.g. BTC, or lots for MT5)
    risk_amount: float    # capital at risk (in quote currency, e.g. USDT/GBP)


def build_trade_plan(
    side: str,
    entry_price: float,
    atr: float,
    equity: float,
    risk_cfg,
) -> TradePlan:
    stop_distance = atr * risk_cfg.atr_stop_multiplier
    risk_amount = equity * (risk_cfg.risk_per_trade_pct / 100)

    if side == "long":
        stop_price = entry_price - stop_distance
        target_price = entry_price + stop_distance * risk_cfg.reward_risk_ratio
    else:
        stop_price = entry_price + stop_distance
        target_price = entry_price - stop_distance * risk_cfg.reward_risk_ratio

    # position size = risk amount / distance to stop
    position_size = risk_amount / stop_distance if stop_distance > 0 else 0.0

    return TradePlan(
        side=side,
        entry_price=entry_price,
        stop_price=stop_price,
        target_price=target_price,
        position_size=position_size,
        risk_amount=risk_amount,
    )
