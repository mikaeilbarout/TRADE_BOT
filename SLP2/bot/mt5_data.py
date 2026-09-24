"""Validated UTC snapshots. API errors never mean an empty account.

MetaTrader5 reports bar, tick, deal and position times as broker *server*
wall-clock seconds (commonly UTC+2/UTC+3 with DST), not UTC. Everything here
converts them to naive UTC; broker range queries are shifted the other way.
"""
import logging
import math
import os
import time
import MetaTrader5 as mt5
import pandas as pd
from bot import config
from bot.bars import utc_naive, validate_bars

log = logging.getLogger("slp2")
TIMEFRAMES = {"M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15, "H1": mt5.TIMEFRAME_H1}
MINUTES = {"M5": 5, "M15": 15, "H1": 60}
CLOCK_SKEW_SECONDS = 2
OFFSET_CONFIRM_SECONDS = 300  # a new offset (DST switch) must persist this long
MAX_OFFSET_HOURS = 14
_clock = {"offset": None, "candidate": None, "since": None}


def utc_now():
    return pd.Timestamp.now(tz="UTC").tz_localize(None)


def reset_server_clock(offset_seconds=None):
    _clock.update(offset=offset_seconds, candidate=None, since=None)


def _fixed_offset():
    """Optional MT5_SERVER_UTC_OFFSET_HOURS pins the offset instead of detecting it."""
    raw = os.getenv("MT5_SERVER_UTC_OFFSET_HOURS", "").strip()
    if not raw:
        return None
    hours = float(raw)
    if not math.isfinite(hours) or abs(hours) > MAX_OFFSET_HOURS:
        raise ValueError("MT5_SERVER_UTC_OFFSET_HOURS must be a finite hour offset")
    return int(round(hours * 3600))


def observe_server_offset(tick, now=None):
    """Whole-hour server-minus-UTC offset implied by a fresh tick, else None.

    A tick is never newer than now, so rounding up to the hour recovers the
    offset while any staleness shows up as residual age and is rejected.
    """
    now = time.time() if now is None else now
    diff = tick.time_msc / 1000 - now
    hours = math.ceil((diff - CLOCK_SKEW_SECONDS) / 3600)
    age = hours * 3600 - diff
    if abs(hours) > MAX_OFFSET_HOURS or not -CLOCK_SKEW_SECONDS <= age <= config.MAX_TICK_AGE_SECONDS:
        return None
    return hours * 3600


def _adopt(observed, now):
    if observed is None:
        return
    if _clock["offset"] is None:
        _clock.update(offset=observed, candidate=None, since=None)
        log.info("Broker server clock is UTC%+.1fh", observed / 3600)
    elif observed == _clock["offset"]:
        _clock.update(candidate=None, since=None)
    elif observed != _clock["candidate"]:
        _clock.update(candidate=observed, since=now)
    elif now - _clock["since"] >= OFFSET_CONFIRM_SECONDS:
        log.warning("Broker server offset changed UTC%+.1fh -> UTC%+.1fh",
                    _clock["offset"] / 3600, observed / 3600)
        _clock.update(offset=observed, candidate=None, since=None)


def refresh_server_offset(symbol, now=None):
    """Update the offset from the live quote; True once it is known."""
    fixed = _fixed_offset()
    if fixed is not None:
        _clock.update(offset=fixed, candidate=None, since=None)
        return True
    tick = mt5.symbol_info_tick(symbol)
    if tick is not None:
        now = time.time() if now is None else now
        _adopt(observe_server_offset(tick, now), now)
    return _clock["offset"] is not None


def server_offset_seconds():
    fixed = _fixed_offset()
    if fixed is not None:
        return fixed
    if _clock["offset"] is None:
        raise RuntimeError("Broker server UTC offset unknown; waiting for a fresh live tick")
    return _clock["offset"]


def server_ms_to_utc(msc):
    return pd.Timestamp(int(msc) - server_offset_seconds() * 1000, unit="ms")


def server_seconds_to_utc_epoch(seconds):
    return seconds - server_offset_seconds()


def utc_to_server_datetime(value):
    """Datetime argument for MT5 range queries (history_deals_get, copy_rates_range)."""
    shifted = utc_naive(value) + pd.Timedelta(seconds=server_offset_seconds())
    return shifted.tz_localize("UTC").to_pydatetime()


def account():
    value = mt5.account_info()
    if value is None:
        raise RuntimeError(f"account_info: {mt5.last_error()}")
    return value


def identity():
    acc = account()
    return f"{acc.server}:{acc.login}:{acc.currency}"


def get_symbol_info(symbol):
    value = mt5.symbol_info(symbol)
    if value is None:
        raise RuntimeError(f"symbol_info {symbol}: {mt5.last_error()}")
    return value


def get_tick(symbol, now=None):
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        raise RuntimeError(f"No tick for {symbol}")
    now = time.time() if now is None else now
    if _fixed_offset() is None:
        _adopt(observe_server_offset(tick, now), now)
    age = now - server_seconds_to_utc_epoch(tick.time_msc / 1000)
    if not (-CLOCK_SKEW_SECONDS <= age <= config.MAX_TICK_AGE_SECONDS):
        raise RuntimeError(f"Stale/future tick for {symbol}: age={age:.1f}s")
    if not all(math.isfinite(x) and x > 0 for x in (tick.bid, tick.ask)) or tick.ask < tick.bid:
        raise RuntimeError(f"Invalid quote for {symbol}")
    return tick


def positions(symbol=None):
    result = mt5.positions_get(**({"symbol": symbol} if symbol else {}))
    if result is None:
        raise RuntimeError(f"positions_get: {mt5.last_error()}")
    return result


def rates_frame(rates, timeframe, asof, offset_seconds=None):
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"No {timeframe} bars")
    offset = server_offset_seconds() if offset_seconds is None else offset_seconds
    frame = pd.DataFrame(rates).rename(columns={"time": "bar_time"})
    frame["bar_time"] = pd.to_datetime(frame["bar_time"] - offset, unit="s").astype("datetime64[ns]")
    frame = validate_bars(frame, sort=True)
    frame = frame[frame.bar_time + pd.Timedelta(minutes=MINUTES[timeframe]) <= utc_naive(asof)]
    return frame.reset_index(drop=True)


def get_bars(symbol, timeframe, count, asof=None):
    if not isinstance(count, int) or count <= 0:
        raise ValueError("count must be a positive integer")
    asof = utc_now() if asof is None else utc_naive(asof)
    rates = mt5.copy_rates_from_pos(symbol, TIMEFRAMES[timeframe], 0, count + 2)
    frame = rates_frame(rates, timeframe, asof).tail(count).reset_index(drop=True)
    if len(frame) < count:
        raise RuntimeError(f"Insufficient history {symbol}/{timeframe}: {len(frame)}/{count}")
    return frame


def get_history(symbol, timeframe, start, end):
    rates = mt5.copy_rates_range(symbol, TIMEFRAMES[timeframe],
        utc_to_server_datetime(start), utc_to_server_datetime(end))
    return rates_frame(rates, timeframe, utc_naive(end))


def get_account_balance():
    return account().balance
