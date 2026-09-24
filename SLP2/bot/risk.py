"""Sizing in account currency, rounded down without violating the budget."""
import math
from decimal import Decimal, ROUND_FLOOR
import MetaTrader5 as mt5
from bot import config
from bot.mt5_data import get_symbol_info, get_tick, account, positions


def validate_risk(value):
    if not math.isfinite(value) or not 0 < value <= config.MAX_RISK_PCT:
        raise ValueError(f"risk must be finite and in (0, {config.MAX_RISK_PCT}]")
    return value


def profit(symbol, direction, lots, entry, exit_price):
    if direction not in (-1, 1) or not all(math.isfinite(v) and v > 0 for v in (lots, entry, exit_price)):
        raise ValueError("Invalid PnL inputs")
    value = mt5.order_calc_profit(mt5.ORDER_TYPE_BUY if direction == 1 else mt5.ORDER_TYPE_SELL,
                                 symbol, lots, entry, exit_price)
    if value is None or not math.isfinite(value):
        raise RuntimeError(f"Cannot calculate account-currency PnL for {symbol}")
    return value


def floor_volume(raw, info):
    if not all(math.isfinite(v) and v > 0 for v in (info.volume_step, info.volume_min, info.volume_max)):
        raise ValueError("Invalid broker volume limits")
    if info.volume_max < info.volume_min:
        raise ValueError("Inverted broker volume limits")
    if not math.isfinite(raw) or raw <= 0:
        return 0.0
    step = Decimal(str(info.volume_step))
    lots = (Decimal(str(min(raw, info.volume_max))) / step).to_integral_value(rounding=ROUND_FLOOR) * step
    return float(lots) if lots >= Decimal(str(info.volume_min)) else 0.0


def lots_for_risk(symbol, stop_dist_price, risk_pct, *, direction=1, entry=None,
                  stop=None, equity=None, available_risk=None):
    validate_risk(risk_pct)
    if not math.isfinite(stop_dist_price) or stop_dist_price <= 0:
        raise ValueError("Stop distance must be positive and finite")
    info = get_symbol_info(symbol)
    if equity is None:
        equity = account().equity
    if not math.isfinite(equity) or equity <= 0:
        return 0.0
    if entry is None:
        tick = get_tick(symbol)
        entry = tick.ask if direction == 1 else tick.bid
    if stop is None:
        stop = entry - direction * stop_dist_price
    if not all(math.isfinite(v) and v > 0 for v in (entry, stop)) or direction not in (-1, 1) or direction * (entry - stop) <= 0:
        raise ValueError("Invalid stop side")
    unit_loss = -profit(symbol, direction, 1.0, entry, stop) + config.COMMISSION_PER_LOT
    if unit_loss <= 0:
        raise ValueError("Invalid loss estimate")
    budget = equity * risk_pct
    if available_risk is not None:
        if not math.isfinite(available_risk):
            return 0.0
        budget = min(budget, available_risk)
    return floor_volume(max(0, budget) / unit_loss, info)


def portfolio_risk():
    total = 0.0
    for pos in positions():
        if not pos.sl:
            return float("inf")
        d = 1 if pos.type == mt5.ORDER_TYPE_BUY else -1
        total += max(0.0, -profit(pos.symbol, d, pos.volume, pos.price_current, pos.sl))
        total += config.COMMISSION_PER_LOT * pos.volume
    return total
