"""
XAUUSD live trading bot on MetaTrader5 (Pepperstone).

WARNING: this script only runs on Windows (the MetaTrader5 package talks to a
local MT5 terminal). It must run on a machine/VPS where:
  1. The MT5 terminal is installed and logged into the Pepperstone demo account
  2. AutoTrading (button at the top of MT5) is enabled
  3. Python and the MetaTrader5 package are installed: pip install MetaTrader5

Usage:
    python mt5/live_bot_mt5.py --profile m1     (or m5 / m15)

For 24/7 operation, wrap this in a Windows Task or NSSM service so it restarts
automatically if the system reboots or the script crashes (see SETUP_MT5.md).
"""

import sys
import os
import time
import math
import logging
import argparse
import msvcrt
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import MetaTrader5 as mt5
except ImportError:
    print("MetaTrader5 package not installed. On Windows run: pip install MetaTrader5")
    sys.exit(1)

import pandas as pd
import importlib

from mt5.config_mt5 import MT5, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
from mt5 import notifier
from mt5.ai_review import review_signal, report_outcome, log_agent_detail  # after config_mt5 loaded .env
from strategy.signals import add_indicators, trend_direction, entry_signal
from strategy.donchian import (
    add_donchian_indicators, add_trend_indicator, add_pivot_trend,
    trend_direction as donchian_trend_direction, donchian_signal,
)
from strategy.risk import build_trade_plan
from mt5.spx_filter import allows_entry as spx_allows_entry, decision_time as spx_decision_time, spx_move_pct

parser = argparse.ArgumentParser()
parser.add_argument("--profile", choices=["m1", "m15", "m30", "h1"], default="m1",
                     help="Which strategy profile to run (default: m1)")
args = parser.parse_args()

profile = importlib.import_module(f"mt5.profiles.profile_{args.profile}")
RISK = profile.RISK
# Two algorithms live side by side now (2026-09-05): the original EMA+RSI+MACD
# mean-reversion approach, and Donchian Channel Breakout -- found to
# dramatically outperform RSI+MACD on most (not all) timeframes when tested
# on 5 years of data (see profile docstrings for exact numbers; M1 in
# particular FAILED with realistic spread despite a huge no-cost backtest
# number, so it's staying on RSI+MACD). Each profile module declares which
# one it uses via ALGORITHM; STRATEGY only exists for "rsi_macd" profiles.
ALGORITHM = getattr(profile, "ALGORITHM", "rsi_macd")
# Skip a signal whose stop is closer than this many USD/oz (0 = off); see profile_m15.py.
MIN_STOP_DOLLARS = getattr(profile, "MIN_STOP_DOLLARS", 0.0)
# Skip a trade when the S&P 500 moved with it over the last 5 trading days (None = off); see profile_m15.py.
SPX_FILTER = getattr(profile, "SPX_FILTER", None)
if ALGORITHM == "rsi_macd":
    STRATEGY = profile.STRATEGY

os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(
            f"logs/mt5_bot_{args.profile}_{datetime.utcnow().strftime('%Y%m%d')}.log",
            encoding="utf-8",
        ),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(f"xauusd_bot_{args.profile}")
log.info(f"Selected profile: {args.profile.upper()} | Magic={profile.MAGIC_NUMBER}")

TF_MAP_LOWER = profile.ENTRY_TIMEFRAME
TF_MAP_HIGHER = profile.TREND_TIMEFRAME

# how long to back off between order-placement retries after a rejection (e.g.
# AutoTrading disabled), instead of retrying every ~10s poll cycle indefinitely
ENTRY_RETRY_COOLDOWN_SECONDS = 60


def acquire_single_instance_lock():
    """
    Prevents two processes for the same profile running at once (e.g. the
    watchdog script and a manual launch both active) -- without this, both
    could see `position is None` at the same moment and both open a
    position under the same magic number. Uses an OS-level exclusive file
    lock (released automatically if the process dies, even a crash), not a
    PID check -- more reliable on Windows than trying to detect a stale PID.
    Keeps the returned file handle open for the life of the process; do not
    let it get garbage-collected.
    """
    lock_path = f"logs/{args.profile}.lock"
    lock_file = open(lock_path, "w")
    try:
        msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        log.error(
            f"Another instance of profile '{args.profile}' appears to already be running "
            f"(lock file: {lock_path}). Exiting rather than risk duplicate positions."
        )
        sys.exit(1)
    return lock_file


def connect(exit_on_fail=True) -> bool:
    if not mt5.initialize(login=MT5.login, password=MT5.password, server=MT5.server):
        log.error(f"MT5 connection failed: {mt5.last_error()}")
        if exit_on_fail:
            sys.exit(1)
        return False
    account_info = mt5.account_info()
    if account_info is None:
        log.error("Could not retrieve account info.")
        if exit_on_fail:
            sys.exit(1)
        return False
    log.info(
        f"Connected to account {account_info.login} | server={account_info.server} "
        f"| balance={account_info.balance:.2f} {account_info.currency} "
        f"| account type={'DEMO' if account_info.trade_mode == 0 else 'LIVE'}"
    )
    if account_info.trade_mode != 0:
        log.warning("This is NOT a demo account! Stop now if that wasn't intended.")
    return True


def is_connected() -> bool:
    """Checks whether the MT5 terminal connection (and internet) is still up."""
    terminal_info = mt5.terminal_info()
    if terminal_info is None:
        return False
    return terminal_info.connected


def reconnect_with_backoff(max_wait_seconds=300):
    """Retries the connection with exponential backoff (capped at 5 minutes)."""
    wait = 5
    attempt = 1
    while True:
        log.warning(f"Connection lost — reconnect attempt {attempt} in {wait}s...")
        time.sleep(wait)
        mt5.shutdown()
        if connect(exit_on_fail=False):
            log.info("Reconnected successfully.")
            return
        attempt += 1
        wait = min(wait * 2, max_wait_seconds)


def fetch_rates(timeframe, count=300) -> pd.DataFrame:
    """
    Returns the last `count` CLOSED bars. mt5.copy_rates_from_pos's position 0
    is the CURRENT, still-forming bar (its "close" is just the live price at
    the moment of the call, not a final value) -- fetched here and dropped so
    every caller's `.iloc[-1]` is a real closed bar, matching what
    strategy/donchian.py::simulate_donchian sees in backtests (which only
    ever has fully-closed historical bars, since that's all export_history.py
    can pull). Bug found 2026-09-09: before this fix, every signal check
    (donchian breakout, trend EMA, RSI/MACD) ran on the live/still-moving
    price of the forming bar instead of a confirmed close -- verified
    empirically: the "last" M1 bar's close exactly matched the current tick's
    bid and had visibly lower volume than fully-formed bars. Most consequential
    for M15/M30/H1 (up to a 15/30/60-minute-early, not-yet-confirmed signal);
    negligible for M1 (1-minute window).
    """
    rates = mt5.copy_rates_from_pos(MT5.symbol, timeframe, 0, count + 1)
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"Failed to fetch {MT5.symbol} data: {mt5.last_error()}")
    df = pd.DataFrame(rates)
    df["ts"] = pd.to_datetime(df["time"], unit="s")
    df = df.rename(columns={"tick_volume": "volume"})
    df = df.iloc[:-1].reset_index(drop=True)  # drop the current, still-forming bar
    return df[["ts", "open", "high", "low", "close", "volume"]]


def get_equity() -> float:
    account_info = mt5.account_info()
    return account_info.equity


_server_offset = None  # seconds: broker server clock minus UTC, learned from live ticks


def server_utc_offset():
    """MT5 stamps ticks and positions in broker SERVER time (e.g. UTC+3), not UTC.

    Learns the whole-hour offset from a fresh tick (same method as SLP2's
    bot/mt5_data.py): a tick is never newer than now, so rounding up to the hour
    recovers the offset and staleness shows up as residual age. Returns None until a
    fresh tick has been seen (e.g. market closed). MT5_SERVER_UTC_OFFSET_HOURS pins it.
    """
    global _server_offset
    fixed = os.getenv("MT5_SERVER_UTC_OFFSET_HOURS", "").strip()
    if fixed:
        return int(round(float(fixed) * 3600))
    tick = mt5.symbol_info_tick(MT5.symbol)
    if tick is not None:
        diff = tick.time_msc / 1000 - time.time()
        hours = math.ceil((diff - 2) / 3600)
        age = hours * 3600 - diff
        if abs(hours) <= 14 and -2 <= age <= 60 and _server_offset != hours * 3600:
            _server_offset = hours * 3600
            log.info(f"Broker server clock is UTC{hours:+d}h")
    return _server_offset


def get_open_position():
    positions = mt5.positions_get(symbol=MT5.symbol)
    if not positions:
        return None
    for p in positions:
        if p.magic == profile.MAGIC_NUMBER:
            return p
    return None


AI_REJECTED = "ai_rejected"
SMALL_STOP = "small_stop"  # signal skipped: stop closer than MIN_STOP_DOLLARS
SPX_BLOCKED = "spx_blocked"  # signal skipped: S&P 500 moved in the same direction (SPX_FILTER)
_spx_cache = {}  # signal bar time -> S&P 500 move %, fetched once per bar


def spx_move_for(signal_bar_ts):
    """S&P 500 % move over SPX_FILTER['lookback_hours'] up to the signal bar's close, from closed H1 bars
    (same calculation as the backtest, see mt5/spx_filter.py). None if the data is unavailable -- the
    filter then lets the trade through. Fetched once per signal bar, so the 3 s loop does not refetch."""
    key = str(signal_bar_ts)
    if key in _spx_cache:
        return _spx_cache[key]
    value = None
    try:
        sym = SPX_FILTER["symbol"]
        if not mt5.symbol_select(sym, True):
            raise RuntimeError(f"symbol_select({sym}) failed: {mt5.last_error()}")
        rates = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_H1, 0, 400)
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"no {sym} H1 data: {mt5.last_error()}")
        times = pd.to_datetime(rates["time"], unit="s")
        value = spx_move_pct(times.values, rates["close"], spx_decision_time(pd.Timestamp(signal_bar_ts).to_pydatetime()),
                             SPX_FILTER["lookback_hours"])
        if value is None:
            log.warning(f"S&P 500 filter: not enough {sym} history -- filter skipped for this bar")
    except Exception as exc:
        log.warning(f"S&P 500 filter: data unavailable ({exc}) -- filter skipped for this bar")
    _spx_cache.clear()
    _spx_cache[key] = value
    return value
AI_SIGNAL_IDS = {}  # position ticket -> AI signal_id, for outcome reports (in-memory only)


def place_order(plan, loss_streak=0):
    symbol_info = mt5.symbol_info(MT5.symbol)
    if symbol_info is None:
        log.error(f"Symbol {MT5.symbol} not found.")
        return None
    if not symbol_info.visible:
        mt5.symbol_select(MT5.symbol, True)

    order_type = mt5.ORDER_TYPE_BUY if plan.side == "long" else mt5.ORDER_TYPE_SELL
    price = mt5.symbol_info_tick(MT5.symbol).ask if plan.side == "long" else mt5.symbol_info_tick(MT5.symbol).bid

    # snap volume to the broker's allowed step, then clamp to [volume_min, volume_max]
    # plan.position_size assumes 1 unit = 1 unit of the underlying price move
    # (correct for crypto spot). MT5 lots represent a contract_size multiple
    # of the underlying (e.g. 1 standard XAUUSD lot = 100 oz), so a $1 price
    # move on 1 lot is worth $contract_size, not $1. Convert accordingly —
    # without this, computed volume is inflated by a factor of contract_size.
    contract_size = symbol_info.trade_contract_size or 1.0
    lots = plan.position_size / contract_size

    step = symbol_info.volume_step or 0.01
    volume = round(lots / step) * step
    volume = round(volume, 2)
    volume = max(symbol_info.volume_min, min(volume, symbol_info.volume_max))
    fixed_lots = getattr(profile, "FIXED_LOTS", None)  # fixed volume per trade (user request 2026-10-04); None = risk-based
    if fixed_lots:
        volume = max(symbol_info.volume_min, min(round(round(fixed_lots / step) * step, 2), symbol_info.volume_max))

    if abs(volume - plan.position_size) > 1e-9:
        log.info(
            f"Position size adjusted to broker limits: risk-based={plan.position_size:.4f} "
            f"| contract_size={contract_size} -> lots={lots:.4f} -> sent={volume} "
            f"(min={symbol_info.volume_min}, max={symbol_info.volume_max}, step={symbol_info.volume_step})"
        )

    # AI review (fail-open, same service and contract as SLP2): only an explicit
    # rejection blocks; an unreachable service lets the trade through.
    account_info = mt5.account_info()
    review = review_signal(
        symbol=MT5.symbol, side=plan.side, entry=price, stop_loss=plan.stop_price,
        take_profit=plan.target_price, volume=volume, timeframe=args.profile.upper(),
        strategy=f"donchian_{args.profile}", equity=account_info.equity if account_info else 0.0,
        recent_loss_streak=loss_streak, market_open=True,  # a live quote was just read above
    )
    log.info(f"AI review: {review.decision} -- {review.reason}")
    if not review.approved:
        return AI_REJECTED
    log_agent_detail(review.signal_id)
    # The review can take minutes: price the order from a fresh quote, and drop it if
    # the planned stop or target is no longer on the correct side of the market.
    tick = mt5.symbol_info_tick(MT5.symbol)
    if tick is None:
        log.warning("No quote after AI review -- skipping this trade.")
        return None
    price = tick.ask if plan.side == "long" else tick.bid
    d = 1 if plan.side == "long" else -1
    if d * (price - plan.stop_price) <= 0 or d * (plan.target_price - price) <= 0:
        log.warning(f"Price {price:.2f} moved past the planned SL/TP during AI review -- skipping.")
        return None

    # sanity-check available margin before sending. Rather than skipping the
    # trade outright when volatility is unusually low (a tight ATR inflates
    # the risk-based lot size well past what free margin can support), scale
    # the volume DOWN to fit -- margin is linear in volume for CFDs, so this
    # is a direct proportional shrink. The trade still opens, just risking
    # less than the configured risk_per_trade_pct instead of being missed
    # entirely. Only skip if even the broker's minimum lot size doesn't fit.
    # (2026-09-04: live M1 was skipping the same real signal repeatedly for
    # minutes during a low-ATR spell -- required margin ~52k vs ~35k free.)
    MARGIN_SAFETY_BUFFER = 0.95  # leave headroom for the price to move before the order fills
    required_margin = mt5.order_calc_margin(order_type, MT5.symbol, volume, price)
    account_info = mt5.account_info()
    if required_margin is None or account_info is None:
        log.warning("Could not verify margin requirements — skipping this trade to be safe.")
        return None
    if required_margin > account_info.margin_free:
        affordable_volume = volume * (account_info.margin_free * MARGIN_SAFETY_BUFFER) / required_margin
        affordable_volume = round(affordable_volume / step) * step
        affordable_volume = round(affordable_volume, 2)

        if affordable_volume < symbol_info.volume_min:
            log.warning(
                f"Skipping trade: even the minimum lot size ({symbol_info.volume_min}) needs more "
                f"margin than available ({account_info.margin_free:.2f} free)."
            )
            return None

        scaled_margin = mt5.order_calc_margin(order_type, MT5.symbol, affordable_volume, price)
        log.warning(
            f"Volume reduced for margin: risk-based {volume} lots needed {required_margin:.2f} margin "
            f"({account_info.margin_free:.2f} free) -> scaled to {affordable_volume} lots "
            f"({scaled_margin:.2f} margin). This trade risks less than the configured "
            f"{RISK.risk_per_trade_pct}% of equity."
        )
        volume = affordable_volume
        required_margin = scaled_margin

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": MT5.symbol,
        "volume": volume,
        "type": order_type,
        "price": price,
        "sl": plan.stop_price,
        "tp": plan.target_price,
        "deviation": MT5.deviation,
        "magic": profile.MAGIC_NUMBER,
        "comment": f"scalp-{args.profile}",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        log.error(f"Order rejected: retcode={result.retcode} | {result.comment}")
        return None
    log.info(
        f"Position opened: {plan.side} {volume} lots {MT5.symbol} @ {price:.2f} "
        f"| SL={plan.stop_price:.2f} TP={plan.target_price:.2f}"
    )
    if review.signal_id:
        AI_SIGNAL_IDS[result.order] = review.signal_id

    equity = get_equity()
    notifier.notify_trade_opened(
        TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,
        profile=args.profile, symbol=MT5.symbol, side=plan.side,
        volume=volume, entry_price=price, stop_price=plan.stop_price,
        target_price=plan.target_price, risk_amount=plan.risk_amount, equity=equity,
    )
    return result


def close_position(position, reason):
    order_type = mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
    price = mt5.symbol_info_tick(MT5.symbol).bid if position.type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(MT5.symbol).ask
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": MT5.symbol,
        "volume": position.volume,
        "type": order_type,
        "position": position.ticket,
        "price": price,
        "deviation": MT5.deviation,
        "magic": profile.MAGIC_NUMBER,
        "comment": f"scalp-{args.profile}-close-{reason}",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        log.error(f"Failed to close position: retcode={result.retcode} | {result.comment}")
    else:
        log.info(f"Position closed ({reason}) | profit={position.profit:.2f}")
        side = "long" if position.type == mt5.ORDER_TYPE_BUY else "short"
        report_outcome(signal_id=AI_SIGNAL_IDS.pop(position.ticket, None), profit=position.profit,
                       exit_reason=reason, ticket=position.ticket, entry_price=position.price_open,
                       close_price=price, volume=position.volume)
        notifier.notify_trade_closed(
            TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,
            profile=args.profile, symbol=MT5.symbol, side=side,
            volume=position.volume, entry_price=position.price_open,
            close_price=price, profit=position.profit, reason=reason,
            equity=get_equity(),
        )
    return result


# A position closed by hand (terminal, mobile or web) must not be re-opened on the same
# M15 bar: the last closed bar's signal is still valid, so without this the bot re-entered
# ~10 s after the user closed it (seen live 2026-09-24). Entries resume at the next bar.
MANUAL_CLOSE_REASONS = {getattr(mt5, n) for n in ("DEAL_REASON_CLIENT", "DEAL_REASON_MOBILE", "DEAL_REASON_WEB")
                        if hasattr(mt5, n)}
manual_close_block_until = None  # UTC; no new entry before this time


def next_m15_bar(now):
    """Start of the next M15 bar (bars are aligned to 15 minutes on any whole-hour clock)."""
    return now.replace(minute=now.minute - now.minute % 15, second=0, microsecond=0) + timedelta(minutes=15)


def check_auto_closed(last_ticket, last_side, last_volume, last_entry_price):
    """
    Called when a tracked position has disappeared without us explicitly closing it —
    meaning the broker closed it automatically (Stop Loss or Take Profit hit).
    Looks up the closing deal to report the exact price/profit/reason via Telegram.
    Returns the realized profit of the close (float), or None if it couldn't be found.
    """
    try:
        deals = mt5.history_deals_get(position=last_ticket)
        if not deals:
            return None
        close_deal = None
        for d in deals:
            if d.entry == mt5.DEAL_ENTRY_OUT:
                close_deal = d
        if close_deal is None:
            return None

        reason_map = {
            mt5.DEAL_REASON_SL: "stop_loss",
            mt5.DEAL_REASON_TP: "take_profit",
            mt5.DEAL_REASON_EXPERT: "expert/manual",
            **{r: "manual" for r in MANUAL_CLOSE_REASONS},
        }
        reason = reason_map.get(close_deal.reason, "unknown")
        if close_deal.reason in MANUAL_CLOSE_REASONS:
            global manual_close_block_until
            manual_close_block_until = next_m15_bar(datetime.utcnow())
            log.info(f"Closed by hand -- no new entry on this M15 bar (until {manual_close_block_until:%H:%M} UTC)")

        log.info(
            f"Auto-close detected (ticket {last_ticket}): reason={reason} "
            f"| close price={close_deal.price:.2f} | profit={close_deal.profit:.2f}"
        )
        report_outcome(signal_id=AI_SIGNAL_IDS.pop(last_ticket, None), profit=close_deal.profit,
                       exit_reason=reason, ticket=last_ticket, entry_price=last_entry_price,
                       close_price=close_deal.price, volume=last_volume)
        notifier.notify_trade_closed(
            TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,
            profile=args.profile, symbol=MT5.symbol, side=last_side,
            volume=last_volume, entry_price=last_entry_price,
            close_price=close_deal.price, profit=close_deal.profit,
            reason=reason, equity=get_equity(),
        )
        return close_deal.profit
    except Exception as e:
        log.warning(f"Could not look up auto-close details: {e}")
        return None


class ConsecutiveLossGuard:
    """
    Optional (per-profile) circuit breaker: pauses new entries after
    `max_losses` losing trades in a row, until a winning trade breaks the
    streak. Disabled entirely when max_losses <= 0.
    Backtested 2026-09-03 (against the RSI+MACD strategy, before the switch
    to Donchian): helped M5/M15's risk-adjusted return slightly but hurt M1's.

    Update 2026-09-04 (streak no longer resets at midnight): a live M5 losing
    streak spanning 2026-09-02 21:45 -> 2026-09-03 07:08 (2 losses on one
    calendar day + 3 on the next, all short trades stopped out during a
    single ~15h period where the trend filter was slow to flip) never
    triggered this breaker, because the old code zeroed consecutive_losses at
    every UTC day boundary -- 2 and 3 each stayed under the threshold of 4
    even though it was really one continuous 5-loss streak.

    Update 2026-09-04 (pause-forever deadlock found and reverted): briefly
    tried removing the daily reset from the pause too (not just the
    counter), so the pause would only clear on an actual winning trade.
    That is a deadlock -- while paused, no new trades open at all, so the
    win needed to clear the pause can never happen. Backtested: M5 dropped
    from 1269 trades/Calmar 17.23 to 6 trades/Calmar 0.22 over 5 years (the
    bot effectively died after its first bad streak). Reverted: the streak
    *counter* still persists across midnight (the actual fix for the
    2026-09-02/03 event), but the *pause* itself still clears at the next
    calendar day, same as before this update -- this is the release valve
    that lets the bot try again instead of freezing forever. This is
    intentionally independent of DailyLossGuard below, which is a genuinely
    calendar-day-scoped guard and is unaffected either way.
    """
    def __init__(self, max_losses: int):
        self.max_losses = max_losses
        self.day = datetime.utcnow().date()
        self.consecutive_losses = 0
        self.paused_today = False

    def _reset_pause_if_new_day(self):
        today = datetime.utcnow().date()
        if today != self.day:
            self.day = today
            self.paused_today = False

    def record_result(self, profit):
        self._reset_pause_if_new_day()
        if self.max_losses <= 0 or profit is None:
            return
        if profit > 0:
            self.consecutive_losses = 0
            self.paused_today = False
        else:
            self.consecutive_losses += 1
            if self.consecutive_losses >= self.max_losses:
                self.paused_today = True
                log.warning(
                    f"{self.consecutive_losses} losses in a row — pausing new entries for the rest of the day."
                )

    def is_paused(self) -> bool:
        self._reset_pause_if_new_day()
        return self.max_losses > 0 and self.paused_today


class CooldownGuard:
    """
    Pauses new entries for a FIXED number of hours after `losses_to_trigger`
    losing trades in a row -- real wall-clock time from the losing exit,
    NOT tied to calendar-day boundaries like ConsecutiveLossGuard above
    (which pauses "for the rest of the day" instead). Disabled when
    losses_to_trigger <= 0.

    Added 2026-09-08 after two straight days of live losing streaks
    (2026-09-07/08) that widening the stop (see profile_m15.py's
    2026-09-07 update) didn't prevent. Backtested via
    strategy/donchian.py::simulate_donchian's matching
    cooldown_losses_to_trigger/cooldown_hours params (swept 2-5 losses x
    0-168h cooldown, walk-forward): a short (~4h) cooldown after 5 losses
    in a row was the only combo that held up on both train and test --
    longer cooldowns looked good on train alone but failed on test
    (overfit). See profile_m15.py's docstring for the full numbers.
    """
    def __init__(self, losses_to_trigger: int, cooldown_hours: float):
        self.losses_to_trigger = losses_to_trigger
        self.cooldown_hours = cooldown_hours
        self.consecutive_losses = 0
        self.cooldown_until = None

    def record_result(self, profit):
        if self.losses_to_trigger <= 0 or profit is None:
            return
        if profit > 0:
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
            if self.consecutive_losses >= self.losses_to_trigger:
                self.cooldown_until = datetime.utcnow() + timedelta(hours=self.cooldown_hours)
                self.consecutive_losses = 0
                log.warning(
                    f"{self.losses_to_trigger} losses in a row — pausing new entries until "
                    f"{self.cooldown_until.isoformat()} UTC ({self.cooldown_hours}h)."
                )

    def is_paused(self) -> bool:
        if self.losses_to_trigger <= 0 or self.cooldown_until is None:
            return False
        return datetime.utcnow() < self.cooldown_until


class DailyLossGuard:
    def __init__(self):
        self.day = datetime.utcnow().date()
        self.start_equity = None

    def check(self, equity, max_loss_pct) -> bool:
        """Returns True if trading is allowed; False if the daily loss cap was hit."""
        today = datetime.utcnow().date()
        if today != self.day or self.start_equity is None:
            self.day = today
            self.start_equity = equity
            log.info(f"New day — baseline equity = {equity:.2f}")
        loss_pct = (self.start_equity - equity) / self.start_equity * 100
        return loss_pct < max_loss_pct


def main():
    _lock = acquire_single_instance_lock()  # noqa: F841 -- held for the process lifetime, never read
    connect()
    guard = DailyLossGuard()
    consec_guard = ConsecutiveLossGuard(getattr(RISK, "max_consecutive_losses", 0))
    cooldown_guard = CooldownGuard(getattr(profile, "COOLDOWN_LOSSES_TO_TRIGGER", 0),
                                    getattr(profile, "COOLDOWN_HOURS", 0.0))
    open_time = None
    time_stop_close_failures = 0
    entry_order_failures = 0
    last_entry_failure_time = None
    ai_rejected_key = None  # (signal, bar time) the AI rejected -- not re-asked on the same bar
    small_stop_key = None  # (signal, bar time) skipped for a too-small stop -- logged once per bar
    spx_block_key = None  # (signal, bar time) skipped by the S&P 500 filter -- logged once per bar

    # tracks the last known open position, so we can detect automatic SL/TP closes
    last_ticket = None
    last_side = None
    last_volume = None
    last_entry_price = None

    log.info(f"XAUUSD bot started | profile={args.profile} | "
             + (f"fixed volume={getattr(profile, "FIXED_LOTS", None)} lots" if getattr(profile, "FIXED_LOTS", None) else f"risk per trade={RISK.risk_per_trade_pct}%"))
    server_utc_offset()  # log the broker clock offset now if the market is open

    consecutive_errors = 0
    last_heartbeat = datetime.utcnow()
    heartbeat_interval_minutes = 5

    try:
        while True:
            try:
                if not is_connected():
                    reconnect_with_backoff()
                    continue

                equity = get_equity()
                if equity is None or equity == 0:
                    log.warning("Failed to read equity — checking connection...")
                    reconnect_with_backoff()
                    continue

                # NOTE: intentionally NOT `continue`-ing here when the daily loss cap is
                # hit -- that used to skip the position-tracking code below entirely,
                # so a position that closed (SL/TP) while capped was never detected:
                # no log, no Telegram notification, and the consecutive-loss guard never
                # learned about it (found 2026-09-07 after a real position's close went
                # completely unlogged). The cap now only blocks *new* entries further down.
                daily_loss_ok = guard.check(equity, RISK.max_daily_loss_pct)

                if ALGORITHM == "donchian":
                    df_low = add_donchian_indicators(fetch_rates(TF_MAP_LOWER, 300), profile.N_PERIOD, profile.ATR_PERIOD)
                    df_high = add_trend_indicator(fetch_rates(TF_MAP_HIGHER, 300), profile.EMA_TREND_PERIOD)
                    row = df_low.iloc[-1]
                    prev_row = df_low.iloc[-2]
                    higher_row = df_high.iloc[-1]
                    ema_trend = donchian_trend_direction(higher_row, getattr(profile, "MIN_TREND_STRENGTH_PCT", 0.0))
                    if getattr(profile, "REQUIRE_PIVOT_CONFIRM", False):
                        # see strategy/donchian.py::add_pivot_trend docstring -- only
                        # trade when the EMA trend and the fractal-pivot HH/HL/LH/LL
                        # swing structure (same TREND_TIMEFRAME) agree.
                        df_high = add_pivot_trend(df_high, getattr(profile, "PIVOT_K", 2))
                        pivot_trend = df_high.iloc[-1]["pivot_trend"]
                        trend = ema_trend if ema_trend == pivot_trend else "flat"
                        heartbeat_extra = (
                            f"donchian_high={row['donchian_high']:.2f} | donchian_low={row['donchian_low']:.2f} "
                            f"| ema_trend={ema_trend} | pivot_trend={pivot_trend}"
                        )
                    else:
                        trend = ema_trend
                        heartbeat_extra = f"donchian_high={row['donchian_high']:.2f} | donchian_low={row['donchian_low']:.2f}"
                else:
                    df_low = fetch_rates(TF_MAP_LOWER, 300)
                    df_low = add_indicators(df_low, STRATEGY)
                    df_high = fetch_rates(TF_MAP_HIGHER, 300)
                    df_high = add_indicators(df_high, STRATEGY)

                    row = df_low.iloc[-1]
                    prev_row = df_low.iloc[-2]
                    higher_row = df_high.iloc[-1]
                    trend = trend_direction(higher_row, getattr(STRATEGY, "min_trend_strength_pct", 0.0))
                    heartbeat_extra = f"rsi={row['rsi']:.1f}"

                # periodic heartbeat so it's clear the bot is alive even with no trades
                now = datetime.utcnow()
                if (now - last_heartbeat).total_seconds() >= heartbeat_interval_minutes * 60:
                    position_status = "position open" if get_open_position() else "no open position"
                    cap_status = " | daily loss cap hit -- no new entries" if not daily_loss_ok else ""
                    if cooldown_guard.is_paused():
                        cap_status += f" | cooldown until {cooldown_guard.cooldown_until.isoformat()} UTC -- no new entries"
                    log.info(
                        f"Heartbeat | price={row['close']:.2f} | trend={trend} | {heartbeat_extra} "
                        f"| equity={equity:.2f} | {position_status}{cap_status}"
                    )
                    last_heartbeat = now

                position = get_open_position()

                if position is None:
                    # if we were tracking an open position and it's gone now, and we
                    # didn't explicitly close it below, the broker closed it (SL/TP) --
                    # this must run regardless of daily_loss_ok, see note above
                    if last_ticket is not None:
                        closed_profit = check_auto_closed(last_ticket, last_side, last_volume, last_entry_price)
                        consec_guard.record_result(closed_profit)
                        cooldown_guard.record_result(closed_profit)
                        last_ticket = None
                        open_time = None

                    manual_block = manual_close_block_until is not None and datetime.utcnow() < manual_close_block_until
                    if daily_loss_ok and trend != "flat" and not consec_guard.is_paused() and not cooldown_guard.is_paused() and not manual_block:
                        if ALGORITHM == "donchian":
                            signal = donchian_signal(row, trend)
                        else:
                            signal = entry_signal(row, prev_row, trend, STRATEGY)
                        if not signal:
                            entry_order_failures = 0
                        else:
                            now_utc = datetime.utcnow()
                            cooling_down = (
                                entry_order_failures > 0 and last_entry_failure_time is not None
                                and (now_utc - last_entry_failure_time).total_seconds() < ENTRY_RETRY_COOLDOWN_SECONDS
                            )
                            if not cooling_down and (signal, str(row["ts"])) != ai_rejected_key:
                                plan = build_trade_plan(signal, row["close"], row["atr"], equity, RISK)
                                stop_distance = abs(plan.entry_price - plan.stop_price)
                                if stop_distance < MIN_STOP_DOLLARS:
                                    if small_stop_key != (signal, str(row["ts"])):
                                        small_stop_key = (signal, str(row["ts"]))
                                        log.info(f"Signal skipped: {signal} stop distance ${stop_distance:.2f} "
                                                 f"< minimum ${MIN_STOP_DOLLARS:.2f} (ATR too small for the costs)")
                                    result = SMALL_STOP  # not `continue`: that would skip the loop's sleep
                                elif SPX_FILTER and not spx_allows_entry(spx_move := spx_move_for(row["ts"]), signal, SPX_FILTER["threshold_pct"]):
                                    if spx_block_key != (signal, str(row["ts"])):
                                        spx_block_key = (signal, str(row["ts"]))
                                        log.info(f"Signal skipped: {signal} -- S&P 500 moved {spx_move:+.2f}% over "
                                                 f"{SPX_FILTER['lookback_hours']:.0f}h in the same direction (limit "
                                                 f"{SPX_FILTER['threshold_pct']}%), gold acting as a risk asset")
                                    result = SPX_BLOCKED
                                else:
                                    result = place_order(plan, loss_streak=consec_guard.consecutive_losses)
                                if result in (SMALL_STOP, SPX_BLOCKED):
                                    entry_order_failures = 0
                                elif result == AI_REJECTED:
                                    ai_rejected_key = (signal, str(row["ts"]))
                                    entry_order_failures = 0
                                elif result:
                                    # Use the order_send result directly rather than a follow-up
                                    # get_open_position() query -- if the position closed (e.g. an
                                    # extremely fast stop-out) before that query ran, get_open_position()
                                    # would return None and last_ticket would never be set, silently
                                    # losing track of the trade (no Telegram notification, and the
                                    # consecutive-loss guard never learns about it). result.order is the
                                    # position's own ticket for a newly-opened position (this account
                                    # is hedging-mode, one ticket per position), available immediately
                                    # with no race.
                                    open_time = datetime.utcnow()
                                    last_ticket = result.order
                                    last_side = signal
                                    last_volume = result.volume
                                    last_entry_price = result.price
                                    entry_order_failures = 0
                                else:
                                    # order was rejected or skipped (e.g. AutoTrading disabled in the
                                    # terminal, or insufficient margin) -- without this cooldown the
                                    # loop retried the identical order every ~10s indefinitely, which
                                    # spammed the log/broker without ever succeeding on its own. Back
                                    # off between attempts and alert once (not every retry) so it's
                                    # clear manual intervention may be needed.
                                    entry_order_failures += 1
                                    last_entry_failure_time = now_utc
                                    if entry_order_failures == 3:
                                        log.warning(
                                            f"Order placement has failed {entry_order_failures} times in a "
                                            f"row for a {signal} signal."
                                        )
                                        notifier.notify_error(
                                            TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,
                                            profile=args.profile,
                                            message=(
                                                f"Order placement failed {entry_order_failures}x in a row "
                                                f"for a {signal} signal -- may need manual intervention "
                                                f"(e.g. AutoTrading disabled in the terminal)."
                                            ),
                                        )
                else:
                    last_ticket = position.ticket
                    last_side = "long" if position.type == mt5.ORDER_TYPE_BUY else "short"
                    last_volume = position.volume
                    last_entry_price = position.price_open

                    # if this process just (re)started, or a prior close attempt failed
                    # and cleared open_time, fall back to the broker's own recorded open
                    # time rather than treating the position as freshly opened (0 min)
                    # position.time is broker SERVER time: convert with the learned offset.
                    # Until it is known (e.g. market closed) the time_stop check waits.
                    if open_time is None:
                        offset = server_utc_offset()
                        if offset is not None:
                            open_time = datetime.utcfromtimestamp(position.time - offset)

                    minutes_open = (datetime.utcnow() - open_time).total_seconds() / 60 if open_time else 0.0
                    if minutes_open >= RISK.time_stop_minutes:
                        overdue_minutes = minutes_open - RISK.time_stop_minutes
                        if overdue_minutes > 15:
                            # more than normal 10s-polling granularity can explain -- the bot was
                            # likely offline/reconnecting through part of this position's lifetime,
                            # during which only its original SL/TP protected it, not the time_stop.
                            log.warning(
                                f"time_stop was {overdue_minutes:.0f} min overdue -- bot may have "
                                f"been offline or reconnecting during this window."
                            )
                            notifier.notify_error(
                                TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,
                                profile=args.profile,
                                message=(
                                    f"time_stop close was {overdue_minutes:.0f} min overdue for a "
                                    f"{last_side} position (ticket {position.ticket}) -- possible bot downtime."
                                ),
                            )
                        closed_profit = position.profit
                        result = close_position(position, "time_stop")
                        if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
                            consec_guard.record_result(closed_profit)
                            cooldown_guard.record_result(closed_profit)
                            last_ticket = None
                            open_time = None
                            time_stop_close_failures = 0
                        else:
                            # close failed (e.g. market closed) -- keep tracking this position so
                            # we retry the time-stop close on the next loop. Alert once (not every
                            # 10s) if it keeps failing, since that may need manual intervention.
                            time_stop_close_failures += 1
                            if time_stop_close_failures == 3:
                                retcode = result.retcode if result is not None else "no result"
                                log.warning(f"time_stop close has failed {time_stop_close_failures} times in a row (retcode={retcode}).")
                                notifier.notify_error(
                                    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,
                                    profile=args.profile,
                                    message=(
                                        f"time_stop close failed {time_stop_close_failures}x in a row "
                                        f"(retcode={retcode}) for ticket {position.ticket} -- may need manual intervention."
                                    ),
                                )
                    # take-profit / stop-loss are placed on the order itself and get
                    # executed automatically by the broker

                consecutive_errors = 0
                time.sleep(3)

            except RuntimeError as e:
                # usually means fetch_rates couldn't get data (temporary connection/broker issue)
                consecutive_errors += 1
                log.error(f"Data error ({consecutive_errors} in a row): {e}")
                if consecutive_errors >= 3:
                    reconnect_with_backoff()
                    consecutive_errors = 0
                else:
                    time.sleep(15)

            except Exception as e:
                consecutive_errors += 1
                log.exception(f"Unexpected error ({consecutive_errors} in a row): {e}")
                if consecutive_errors >= 3:
                    reconnect_with_backoff()
                    consecutive_errors = 0
                else:
                    time.sleep(15)

    except KeyboardInterrupt:
        log.info("Bot stopped manually.")
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()
