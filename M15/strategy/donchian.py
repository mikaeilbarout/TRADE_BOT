"""
Donchian Channel Breakout signal logic -- an alternative to signals.py's
EMA+RSI+MACD mean-reversion approach. Added 2026-09-05 after the user asked
why every profile kept reusing RSI+MACD instead of testing genuinely
different algorithms against the 5 years of data available. Compared
against Bollinger mean-reversion and the original RSI+MACD approach on both
BTC/USDT (scalp_alerts project) and XAUUSD -- Donchian won clearly on most
timeframes for both assets (see profile docstrings for exact numbers).

Pure trend-following breakout: go long when price closes above its own
N-period high while the higher-timeframe trend agrees, short on breakdown
below the N-period low. No RSI, no MACD.
"""

import numpy as np
import pandas as pd


def find_pivots(high, low, k: int = 2):
    """5-bar fractal pivot: a bar is a pivot high/low if its high/low is the
    extreme within k bars on each side."""
    win = 2 * k + 1
    roll_max = pd.Series(high).rolling(win, center=True, min_periods=win).max().to_numpy()
    roll_min = pd.Series(low).rolling(win, center=True, min_periods=win).min().to_numpy()
    is_h = high == roll_max
    is_l = low == roll_min
    is_h[np.isnan(roll_max)] = False
    is_l[np.isnan(roll_min)] = False
    return is_h, is_l


def add_pivot_trend(df: pd.DataFrame, pivot_k: int = 2) -> pd.DataFrame:
    """
    Added 2026-09-08, per user request, as a second, independent trend
    confirmation on top of add_trend_indicator's EMA -- requires the
    higher-timeframe swing structure (fractal pivots, chained and labeled
    HH/HL/LH/LL) to agree with the EMA before a signal is allowed, on the
    theory that the EMA can flip "short" on a shallow dip while the
    underlying swing structure (which needs a real lower-high + lower-low to
    flip) hasn't actually broken down -- exactly what happened 2026-09-07/08:
    5 losing M15/M5 trades in a row where ema_trend said "short" but
    pivot_trend was still "long"/"flat".

    IMPORTANT, tested via full walk-forward backtest before this was wired
    in live (see chat): across the full ~4.2-year M15 history this filter
    LOWERS the walk-forward score (23.08 -> 16.27) and full-dataset return
    (removes 250 of 762 trades net +$10,274 profitable as a group, not net
    losing) -- it only would have prevented this one specific whipsaw
    stretch, not improved things on average. Wired in anyway per explicit
    user request (2026-09-08), accepting the average-case cost to avoid
    this specific failure mode. Reduces trade frequency ~762->512 over the
    same period (~1 trade/2 days -> ~1 trade/3 days).

    A pivot is confirmed `pivot_k` bars after it forms (needs that many bars
    on each side to know it's the local extreme) -- no lookahead.
    """
    df = df.copy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    n = len(df)
    is_h, is_l = find_pivots(high, low, pivot_k)

    chain_last_type = None
    last_h = prev_h = np.nan
    last_l = prev_l = np.nan
    trend = np.full(n, "flat", dtype=object)

    for i in range(n):
        ci = i - pivot_k
        if ci >= 0:
            if is_h[ci]:
                v = high[ci]
                if chain_last_type == "high":
                    if v > last_h:
                        last_h = v
                else:
                    prev_h, last_h = last_h, v
                    chain_last_type = "high"
            if is_l[ci]:
                v = low[ci]
                if chain_last_type == "low":
                    if v < last_l:
                        last_l = v
                else:
                    prev_l, last_l = last_l, v
                    chain_last_type = "low"
        if not (np.isnan(prev_h) or np.isnan(last_h) or np.isnan(prev_l) or np.isnan(last_l)):
            hh = last_h > prev_h
            hl = last_l > prev_l
            lh = last_h < prev_h
            ll = last_l < prev_l
            if hh and hl:
                trend[i] = "long"
            elif lh and ll:
                trend[i] = "short"

    df["pivot_trend"] = trend
    return df


def add_donchian_indicators(df: pd.DataFrame, n_period: int, atr_period: int = 14) -> pd.DataFrame:
    df = df.copy()
    df["donchian_high"] = df["high"].rolling(n_period).max().shift(1)
    df["donchian_low"] = df["low"].rolling(n_period).min().shift(1)

    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = tr.rolling(atr_period).mean()
    return df


def add_trend_indicator(df: pd.DataFrame, ema_period: int) -> pd.DataFrame:
    """Higher-timeframe trend filter: price above/below a single EMA (not a
    fast/slow crossover like the RSI+MACD strategy uses)."""
    df = df.copy()
    df["ema_trend"] = df["close"].ewm(span=ema_period, adjust=False).mean()
    return df


def trend_direction(higher_tf_row, min_strength_pct: float = 0.0) -> str:
    """
    min_strength_pct (default 0 = old behavior): requires price to be at
    least this % away from the trend EMA before trusting the trend, treating
    a marginal/borderline crossing as "flat" instead. Added 2026-09-06 after
    a loss-pattern study found win-vs-loss trades differed most consistently
    (4 of 5 profiles) on this exact feature -- trades taken with price
    already well clear of the EMA won more often than ones taken right at
    the line. NOT yet proven to improve Calmar once actually backtested --
    see profile docstrings for the per-profile verdict.
    """
    if pd.isna(higher_tf_row["ema_trend"]):
        return "flat"
    if min_strength_pct > 0:
        dist_pct = abs(higher_tf_row["close"] - higher_tf_row["ema_trend"]) / higher_tf_row["close"] * 100
        if dist_pct < min_strength_pct:
            return "flat"
    return "long" if higher_tf_row["close"] > higher_tf_row["ema_trend"] else "short"


def donchian_signal(row, trend: str) -> str | None:
    if pd.isna(row["donchian_high"]) or pd.isna(row["donchian_low"]) or pd.isna(row["atr"]) or row["atr"] == 0:
        return None
    if trend == "long" and row["close"] > row["donchian_high"]:
        return "long"
    if trend == "short" and row["close"] < row["donchian_low"]:
        return "short"
    return None


def simulate_donchian(df_low: pd.DataFrame, df_high: pd.DataFrame, n_period: int, atr_period: int,
                       ema_trend_period: int, atr_stop_multiplier: float, reward_risk_ratio: float,
                       time_stop_minutes: int, risk_cfg, starting_equity: float = 1000.0,
                       spread_dollars: float = 0.0, min_trend_strength_pct: float = 0.0,
                       commission_dollars: float = 0.0, swap_long: float = 0.0, swap_short: float = 0.0,
                       require_pivot_confirm: bool = False, pivot_k: int = 2,
                       df_pivot: pd.DataFrame = None,
                       cooldown_losses_to_trigger: int = 0, cooldown_hours: float = 0.0):
    """
    Donchian-equivalent of the RSI+MACD engine that used to live in
    strategy/backtest_engine.py (removed 2026-09-06 as dead code once every
    profile switched to Donchian) -- same conservative same-candle-stop-wins
    assumption, same real-minutes time stop, same daily-loss-guard support.

    THE canonical Donchian backtest -- matches what mt5/live_bot_mt5.py
    actually does live (including the DailyLossGuard, which an earlier quick
    grid-search script did NOT model, causing a real discrepancy: 2026-09-06,
    M15 showed Calmar 214 without this guard vs 316 with it -- the guard
    turned out to be a net positive, cutting off same-day pile-on losses,
    not just a conservative drag. Use THIS function for any number you plan
    to trust or report -- not a fresh one-off script -- to avoid this again.

    spread_dollars (default 0, i.e. no cost): the FULL round-trip cost per
    unit size (the bid-ask spread itself -- crossing it once on entry and
    once on exit already totals one spread-width, not two), charged ONCE per
    closed trade. XAUUSD estimate used elsewhere in this project: $0.30/oz.
    (Bug 2026-09-06 to 2026-09-09: this used to be charged twice -- see the
    spread_cost line below for the fix and its impact on past numbers.)

    commission_dollars (default 0, added 2026-09-06): round-trip Razor-account
    commission per unit size (this account IS Razor, confirmed live via
    symbol Gold - the account pays commission on top of the raw spread, not
    instead of it). Sourced from Pepperstone's own pricing page: $7.00/lot
    round-trip for XAUUSD on MT5 -> $0.07/oz. No official XAGUSD figure is
    published; the same $7/lot is used as a same-broker proxy -> $0.0014/oz
    (silver's much larger 5000oz lot makes this nearly negligible per-oz).

    swap_long / swap_short (default 0, added 2026-09-06): overnight financing
    cost per unit size PER NIGHT held, charged only on trades that span at
    least one full day (confirmed via live MT5 symbol_info -- this account
    trades spot XAUUSD/XAGUSD CFDs, which unlike a futures-style forward
    (e.g. XAUUSD-F, swap_mode=0 on this account) charge a real, currently
    substantial daily swap: -$79.73/lot/night long, +$29.62/lot/night short
    for gold; -$86.00/lot/night long, +$25.05/lot/night short for silver, as
    of 2026-09-06 (these are live, currently-quoted rates that will drift
    with interest rates -- refresh via mt5.symbol_info(...).swap_long/short
    periodically, don't treat as permanent). This was the single biggest
    previously-unmodeled cost found in this project: -32% to -60% Calmar on
    the longest-holding profiles (M30/H1) once included -- see profile
    docstrings and README.md for the full before/after per profile.
    Approximated as floor(minutes_open / 1440) nights -- ignores the
    "triple swap Wednesday" convention many brokers use to charge 3 nights'
    worth on Wednesdays to cover the weekend, so real cost may be somewhat
    higher than this models for trades spanning a Wednesday rollover.

    cooldown_losses_to_trigger / cooldown_hours (added 2026-09-08, default
    0 = disabled): after this many LOSING trades in a row, pause new entries
    for a FIXED `cooldown_hours` (real wall-clock time from the losing exit,
    NOT tied to calendar-day boundaries like live_bot_mt5.py's separate
    ConsecutiveLossGuard, which instead pauses "for the rest of the [UTC]
    day"). Swept for M15 (losses_to_trigger 2-5, cooldown 0-168h,
    train/test walk-forward): a short cooldown (~3-4h) after 5 losses in a
    row was the only combo that beat the no-cooldown baseline on BOTH train
    and test consistently -- score 25.40 -> 26.08, full-dataset max_dd
    -16.0% -> -15.0%, all 4 quarters improved together. Longer cooldowns
    (6h+) looked good on train alone but failed on test (overfit); this
    matches the live-trading finding 2026-09-07/08 that widening the stop
    doesn't stop a real losing streak -- pausing briefly instead does,
    without needing to guess which single trade in the streak was "the"
    mistake.
    """
    import numpy as np

    df_low_i = add_donchian_indicators(df_low, n_period, atr_period)
    df_high_i = add_trend_indicator(df_high, ema_trend_period)
    # vectorized equivalent of trend_direction() applied row-wise (was
    # `df_high_i.apply(lambda r: trend_direction(r, ...), axis=1)` -- ~170x
    # slower for the same result; verified byte-identical output before
    # switching, 2026-09-06 code-quality pass).
    close_h = df_high_i["close"].to_numpy()
    ema_h = df_high_i["ema_trend"].to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        dist_pct = np.abs(close_h - ema_h) / close_h * 100
    ema_trend = np.where(
        pd.isna(ema_h), "flat",
        np.where((min_trend_strength_pct > 0) & (dist_pct < min_trend_strength_pct), "flat",
                 np.where(close_h > ema_h, "long", "short")),
    )
    if require_pivot_confirm:
        # pivot_k confirmation can run on its OWN timeframe (df_pivot, e.g. H1)
        # independent of the EMA's trend timeframe (df_high, e.g. H4) --
        # falls back to df_high itself if no separate df_pivot is given.
        pivot_source = add_pivot_trend(df_pivot if df_pivot is not None else df_high, pivot_k)
        pivot_aligned = pd.merge_asof(
            df_high_i[["ts"]].sort_values("ts"),
            pivot_source[["ts", "pivot_trend"]].sort_values("ts"),
            on="ts", direction="backward",
        )
        pivot_trend = pivot_aligned["pivot_trend"].to_numpy()
        df_high_i["trend"] = np.where((ema_trend == "long") & (pivot_trend == "long"), "long",
                                       np.where((ema_trend == "short") & (pivot_trend == "short"), "short", "flat"))
    else:
        df_high_i["trend"] = ema_trend
    # Fix 2026-09-24 (look-ahead): "ts" is a bar's OPEN time. Matching the M15 bar's open
    # to the latest H4 *open* used the H4 bar still forming at that moment, i.e. its future
    # close (up to 4h ahead) -- which live_bot_mt5.py can never see (fetch_rates drops the
    # forming bar). A trend value is now usable only once its H4 bar has CLOSED, by the
    # time the M15 signal bar closes -- exactly the live bot's view. This alone moved the
    # 4-year M15 result from +95.9% to +14.9% (see research_20260924/).
    low_step = df_low_i["ts"].diff().median()
    high_step = df_high_i["ts"].diff().median()
    known = df_high_i[["ts", "trend"]].assign(_known_at=df_high_i["ts"] + high_step)[["_known_at", "trend"]]
    df_low_i = pd.merge_asof(
        df_low_i.assign(_known_at=df_low_i["ts"] + low_step).sort_values("_known_at"),
        known.sort_values("_known_at"), on="_known_at", direction="backward",
    ).drop(columns="_known_at")

    ts_arr = df_low_i["ts"].to_numpy()
    high = df_low_i["high"].to_numpy(); low = df_low_i["low"].to_numpy(); close = df_low_i["close"].to_numpy()
    dh = df_low_i["donchian_high"].to_numpy(); dl = df_low_i["donchian_low"].to_numpy()
    atr = df_low_i["atr"].to_numpy(); trend = df_low_i["trend"].to_numpy()
    day_arr = pd.Series(ts_arr).dt.date.to_numpy()
    n = len(df_low_i)

    equity = starting_equity
    trades = []
    open_trade = None
    open_bar_idx = None
    current_day = None
    day_start_equity = equity
    daily_loss_paused = False
    max_daily_loss_pct = getattr(risk_cfg, "max_daily_loss_pct", None)
    consec_losses = 0
    cooldown_until = None

    for i in range(n_period + 1, n):
        day = day_arr[i]
        if day != current_day:
            current_day = day
            day_start_equity = equity
            daily_loss_paused = False

        t = trend[i]
        if t is None or (isinstance(t, float) and pd.isna(t)):
            continue

        if open_trade is None:
            if daily_loss_paused:
                continue
            if cooldown_until is not None and ts_arr[i] < cooldown_until:
                continue
            a = atr[i]
            if pd.isna(a) or a == 0 or pd.isna(dh[i]):
                continue
            sig = None
            if t == "long" and close[i] > dh[i]:
                sig = "long"
            elif t == "short" and close[i] < dl[i]:
                sig = "short"
            if sig:
                stop_dist = a * atr_stop_multiplier
                target_dist = stop_dist * reward_risk_ratio
                entry_price = close[i]
                stop_price = entry_price - stop_dist if sig == "long" else entry_price + stop_dist
                target_price = entry_price + target_dist if sig == "long" else entry_price - target_dist
                risk_amount = equity * (risk_cfg.risk_per_trade_pct / 100)
                position_size = risk_amount / stop_dist if stop_dist > 0 else 0.0
                open_trade = {
                    "side": sig, "entry_price": entry_price, "stop_price": stop_price,
                    "target_price": target_price, "position_size": position_size,
                }
                open_bar_idx = i
        else:
            # real wall-clock elapsed time, matching mt5/live_bot_mt5.py's
            # `(datetime.utcnow() - open_time).total_seconds() / 60` exactly.
            # Bug found 2026-09-06: this used to be approximated as
            # `(i - open_bar_idx) * bar_minutes` (bar-count times the
            # DATASET-WIDE MEDIAN bar interval) -- wrong whenever a trade
            # spans a weekend/holiday gap (FX data has real gaps of 2-3 days
            # where the market is closed), since that approximation silently
            # compresses the gap to a single median-sized step instead of its
            # real duration. Materially changed Calmar once fixed: gold M5
            # 23.71->30.27, M30 30.73->41.19 (profiles with many time_stop
            # exits); profiles with few/no time_stop exits were unaffected.
            minutes_open = (ts_arr[i] - ts_arr[open_bar_idx]) / np.timedelta64(1, "m")
            ot = open_trade
            hi_, lo_, cl_ = high[i], low[i], close[i]
            hit_stop = lo_ <= ot["stop_price"] if ot["side"] == "long" else hi_ >= ot["stop_price"]
            hit_target = hi_ >= ot["target_price"] if ot["side"] == "long" else lo_ <= ot["target_price"]
            hit_time = minutes_open >= time_stop_minutes

            if hit_stop or hit_target or hit_time:
                # conservative: stop wins if both touched in the same bar
                if hit_stop:
                    exit_price, outcome = ot["stop_price"], "stop"
                elif hit_target:
                    exit_price, outcome = ot["target_price"], "target"
                else:
                    exit_price, outcome = cl_, "time_stop"

                price_diff = (exit_price - ot["entry_price"]) if ot["side"] == "long" else (ot["entry_price"] - exit_price)
                # spread_dollars is already the FULL round-trip cost (see docstring) -- do
                # NOT multiply by 2. Bug found 2026-09-09: this used to be `* 2`, silently
                # doubling every spread cost in every backtest/tick-verification run this
                # project has ever produced (confirmed live: mt5.symbol_info("XAUUSD").spread
                # -> $0.09/oz observed, well under the modeled $0.30 even without doubling).
                # Effect: every historical Calmar/win-rate/drawdown number this session
                # computed was somewhat MORE conservative than real live economics -- not
                # dangerous, but should be kept in mind when comparing new numbers (now
                # correct) against old ones (quoted with the doubled cost) in past chat/docs.
                spread_cost = ot["position_size"] * spread_dollars
                commission_cost = ot["position_size"] * commission_dollars  # already a round-trip figure
                nights_held = int(minutes_open // 1440)
                swap_rate = swap_long if ot["side"] == "long" else swap_short
                swap_term = swap_rate * ot["position_size"] * nights_held if nights_held > 0 else 0.0
                pnl = price_diff * ot["position_size"] - spread_cost - commission_cost + swap_term
                equity += pnl
                trades.append({
                    "entry_time": ts_arr[open_bar_idx], "exit_time": ts_arr[i], "side": ot["side"],
                    "entry_price": ot["entry_price"], "exit_price": exit_price, "outcome": outcome,
                    "pnl": pnl, "equity_after": equity,
                })
                open_trade = None
                open_bar_idx = None

                if max_daily_loss_pct is not None and day_start_equity > 0:
                    loss_pct = (day_start_equity - equity) / day_start_equity * 100
                    if loss_pct >= max_daily_loss_pct:
                        daily_loss_paused = True

                if cooldown_losses_to_trigger > 0:
                    if pnl < 0:
                        consec_losses += 1
                        if consec_losses >= cooldown_losses_to_trigger:
                            cooldown_until = ts_arr[i] + np.timedelta64(int(cooldown_hours * 60), "m")
                            consec_losses = 0
                    else:
                        consec_losses = 0

    return pd.DataFrame(trades), equity
