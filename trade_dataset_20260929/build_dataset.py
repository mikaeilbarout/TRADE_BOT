"""Dataset of every backtest trade of ONE bot: Donchian M15 with the live settings
(channel 10, H4 EMA30 trend >= 0.5%, stop 2x ATR14 with a $8 minimum, RR 3, risk 0.2%).
One continuous history without overlapping positions: the look-ahead-free bar engine
up to the start of the tick data, then the real-tick simulation.

Output (this folder): donchian_dataset.csv / .parquet + schema.json -- pre-entry features, then the result.
Times: `*_server` / decision_time are broker server time (UTC+3 summer, UTC+2 winter);
`*_utc` columns convert them with the EET/EEST rule (approximate around DST switches).
(slp2_* and live_trades() below are kept for reuse; main() no longer writes them.)
"""
import sys, os, types
HERE = os.path.dirname(os.path.abspath(__file__)); COMB = os.path.dirname(HERE)
M15 = os.path.join(COMB, "M15"); SLP2 = os.path.join(COMB, "SLP2")
for p in (COMB, SLP2, M15, os.path.join(M15, "research_20260924")):
    sys.path.insert(0, p)
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import combined_tick_backtest as ctb                      # imports the real MetaTrader5 package if present
from scripts.pattern_strategy import load_m15
from scripts.sp2l_m15_backtest import simulate as slp2_simulate
from scripts.sp2l_tick_backtest import run_ticks as slp2_run_ticks
from bot.sp2l import DEFAULTS as SLP2_DEFAULTS
from strategy.donchian import add_donchian_indicators, add_trend_indicator
import reevaluate_donchian as rd

EQUITY, RISK = ctb.EQUITY, ctb.RISK_PCT
BAR = pd.Timedelta(minutes=15)
DON = dict(n_period=20, ema_trend_period=30, min_trend_strength_pct=.3, atr_stop_multiplier=2.0, reward_risk_ratio=4.0, min_stop_dollars=8.0)

# ---------------------------------------------------------------- market context
frame = load_m15()
bars = add_donchian_indicators(frame.rename(columns={"bar_time": "ts"})[["ts", "open", "high", "low", "close"]], DON["n_period"], 14)
# activity at the signal bar (tick volume = number of price updates, the only volume MT5 has for gold)
bars["vol"] = frame.tick_volume.to_numpy(float); bars["spread"] = frame.avg_spread_price.to_numpy(float)
bars["vol_vs_20"] = bars.vol / bars.vol.shift(1).rolling(20).mean()
bars["vol4h_vs_5d"] = bars.vol.rolling(16).sum() / (bars.vol.shift(16).rolling(96 * 5).sum() / 30)
bars["spread_vs_5d"] = bars.spread / bars.spread.shift(1).rolling(96 * 5).mean()
bars["atr_ratio"] = bars.atr / bars.atr.rolling(96 * 5).mean()
bars["ema200_m15"] = bars.close.ewm(span=200, adjust=False).mean()
bars = bars.set_index("ts")
h4 = frame.set_index("bar_time")[["open", "high", "low", "close"]].resample("4h", label="left", closed="left").agg(
    {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
h4 = add_trend_indicator(h4, 30)
side_h = np.sign(h4.close - h4.ema_trend)
h4["side"] = side_h
h4["age"] = side_h.groupby((side_h != side_h.shift()).cumsum()).cumcount() + 1
h4["dist_pct"] = (h4.close - h4.ema_trend) / h4.close * 100
h4["slope_pct"] = h4.ema_trend.pct_change(6) * 100
h4 = h4.assign(known=h4.index + pd.Timedelta(hours=4)).reset_index(drop=True).sort_values("known")
d1 = frame.set_index("bar_time").resample("1D").agg({"high": "max", "low": "min", "close": "last"}).dropna()
tr = pd.concat([d1.high - d1.low, (d1.high - d1.close.shift()).abs(), (d1.low - d1.close.shift()).abs()], axis=1).max(axis=1)
up, dn = d1.high.diff(), -d1.low.diff()
pdm, ndm = up.where((up > dn) & (up > 0), 0.), dn.where((dn > up) & (dn > 0), 0.)
a14 = tr.ewm(alpha=1 / 14).mean(); pdi = 100 * pdm.ewm(alpha=1 / 14).mean() / a14; ndi = 100 * ndm.ewm(alpha=1 / 14).mean() / a14
d1["adx"] = (100 * (pdi - ndi).abs() / (pdi + ndi)).ewm(alpha=1 / 14).mean()
d1["ema50"] = d1.close.ewm(span=50, adjust=False).mean()
d1["atr14"] = a14
daily_close = d1.close.copy()                             # raw daily closes (server days), for the day-by-day moves


def _older_daily_closes(before):
    """The M15 file starts 2022-06-27, too late for the first trades' 7-day / 30-day moves:
    fetch the missing daily closes from the broker (read-only MT5, same server clock)."""
    stub = sys.modules.pop("MetaTrader5", None)            # reevaluate_donchian may have installed a stub
    try:
        import MetaTrader5 as real_mt5
        if not hasattr(real_mt5, "copy_rates_range") or not real_mt5.initialize():
            return pd.Series(dtype=float)
        real_mt5.order_send = None
        r = real_mt5.copy_rates_range("XAUUSD", real_mt5.TIMEFRAME_D1, (before - pd.Timedelta(days=60)).to_pydatetime(), before.to_pydatetime())
        real_mt5.shutdown()
    except Exception:
        return pd.Series(dtype=float)
    finally:
        if stub is not None and not hasattr(stub, "copy_rates_range"):
            sys.modules["MetaTrader5"] = stub
    s = pd.DataFrame(r)
    if s.empty:
        return pd.Series(dtype=float)
    s = pd.Series(s.close.values, index=pd.to_datetime(s.time, unit="s"))
    return s[s.index < before]


daily_close = pd.concat([_older_daily_closes(daily_close.index[0]), daily_close]).sort_index()
d1 = d1.shift(1)                                           # only the previous completed day is known
m5 = pd.read_parquet(os.path.join(SLP2, "data", "XAUUSD_M5_full_history.parquet")).set_index("bar_time")


def server_to_utc(ts):
    ts = pd.to_datetime(ts)
    return (ts.dt.tz_localize("Europe/Athens", ambiguous="NaT", nonexistent="shift_forward")
              .dt.tz_convert("UTC").dt.tz_localize(None))


def session(h_utc):
    return pd.cut(h_utc, [-1, 6, 11, 15, 20, 23], labels=["Asia", "London", "London-NY overlap", "New York", "late/close"]).astype(str)


def last_closed_bar(decision_time):
    """Open time of the last M15 bar that had closed at the decision (ts + 15 min <= decision);
    robust to market gaps, where the bar just before the decision may not exist."""
    pos = bars.index.searchsorted((pd.to_datetime(decision_time) - BAR).values, side="right") - 1
    return pd.Series(bars.index[pos], index=getattr(decision_time, "index", None))


def context(t):
    """t needs: decision_time (server clock, when the order is sent), d (+1/-1), entry, stop_dist, R."""
    t = t.sort_values("decision_time").reset_index(drop=True)
    sig_ts = last_closed_bar(t.decision_time)
    s = bars.reindex(sig_ts)
    d = t.d.values
    t["signal_bar_server"] = sig_ts.values
    t["atr14_usd"] = s.atr.values
    t["atr_ratio_5d"] = s.atr_ratio.values
    t["stop_in_atr"] = t.stop_dist.values / s.atr.values
    t["channel10_high"] = s.donchian_high.values; t["channel10_low"] = s.donchian_low.values
    t["channel10_width_atr"] = (s.donchian_high.values - s.donchian_low.values) / s.atr.values
    t["beyond_channel10_atr"] = d * (s.close.values - np.where(d == 1, s.donchian_high.values, s.donchian_low.values)) / s.atr.values
    rng_ = (s.high - s.low).values
    t["signal_close_in_bar_dir"] = np.where(d == 1, (s.close.values - s.low.values), (s.high.values - s.close.values)) / np.where(rng_ > 0, rng_, np.nan)
    t["signal_bar_range_atr"] = rng_ / s.atr.values
    for n, lab in ((4, "1h"), (16, "4h"), (96, "24h")):
        past = bars.close.shift(n).reindex(sig_ts).values
        t[f"move_{lab}_dir_atr"] = d * (s.close.values - past) / s.atr.values
    t["vs_m15_ema200_dir"] = d * np.sign(s.close.values - s.ema200_m15.values)
    k = pd.merge_asof(pd.DataFrame({"known": t.decision_time}), h4, on="known", direction="backward")
    t["h4_trend_with_trade"] = (k.side.values == d).astype(int)
    t["h4_trend_age_bars"] = k.age.values
    t["h4_ema30_dist_pct_dir"] = d * k.dist_pct.values
    t["h4_ema30_slope_1d_pct_dir"] = d * k.slope_pct.values
    # latest daily row at or before the decision day (a decision at 00:00 server time on a
    # Saturday has no row of its own); every row holds the PREVIOUS completed day only
    dd = d1.reindex(t.decision_time.dt.normalize(), method="ffill")
    t["d1_adx14"] = dd.adx.values
    t["d1_with_ema50"] = (np.sign(dd.close.values - dd.ema50.values) == d).astype(int)
    t["d1_atr14_usd"] = dd.atr14.values
    t["hour_server"] = t.decision_time.dt.hour
    t["decision_time_utc"] = server_to_utc(t.decision_time).values
    t["hour_utc"] = pd.to_datetime(t.decision_time_utc).dt.hour
    t["session_utc"] = session(t.hour_utc)
    t["weekday"] = t.decision_time.dt.day_name()
    t["year"] = t.decision_time.dt.year; t["month"] = t.decision_time.dt.strftime("%Y-%m")
    # sequence within this source
    streak, out = 0, []
    for r in t.R:
        out.append(streak); streak = streak + 1 if r <= 0 else 0
    t["prior_loss_streak"] = out
    t["prev_R"] = t.R.shift()
    t["prev_same_side"] = (t.side == t.side.shift()).astype(int)
    t["hours_since_prev_exit"] = (t.decision_time - pd.to_datetime(t.exit_time).shift()).dt.total_seconds() / 3600
    return t


def path_stats(t, src):
    """MFE/MAE during the trade, early behaviour, and what happened after the exit (src = M15 bars or M5 bars)."""
    mfe, mae, early, after = [], [], [], []
    for _, x in t.iterrows():
        seg = src.loc[x.decision_time:pd.Timestamp(x.exit_time)]
        if x.d == 1:
            fav, adv = (seg.high - x.entry).max(), (x.entry - seg.low).max()
        else:
            fav, adv = (x.entry - seg.low).max(), (seg.high - x.entry).max()
        mfe.append(max(0., fav) / x.stop_dist if len(seg) else 0.); mae.append(max(0., adv) / x.stop_dist if len(seg) else 0.)
        nxt = bars.loc[x.decision_time:].iloc[:2]           # first two M15 bars after the entry
        lvl = x.channel10_high if x.d == 1 else x.channel10_low
        early.append(dict(first_bar_back_inside_channel=int(len(nxt) > 0 and x.d * (nxt.close.iloc[0] - lvl) < 0),
                          first_bar_close_move_R=float(x.d * (nxt.close.iloc[0] - x.entry) / x.stop_dist) if len(nxt) else np.nan))
        post = src.loc[pd.Timestamp(x.exit_time):pd.Timestamp(x.exit_time) + pd.Timedelta(hours=24)]
        tgt = x.entry + x.d * x.target_dist
        after.append(int(len(post) > 0 and ((post.high >= tgt).any() if x.d == 1 else (post.low <= tgt).any())))
    t["mfe_R"] = mfe; t["mae_R"] = mae
    t = pd.concat([t, pd.DataFrame(early, index=t.index)], axis=1)
    t["reached_target_within_24h_after_exit"] = after
    t["fake_breakout"] = ((t.R < 0) & (t.mfe_R < .25)).astype(int)
    t["gave_back_1R"] = ((t.R < 0) & (t.mfe_R >= 1)).astype(int)
    return t


def finish(t, source, bot, src):
    t["source"] = source; t["bot"] = bot
    t["side"] = np.where(t.d == 1, "long", "short")
    t = context(t)
    t = path_stats(t, src)
    t["win"] = (t.R > 0).astype(int)
    t["hold_hours"] = (pd.to_datetime(t.exit_time) - t.decision_time).dt.total_seconds() / 3600
    t["exit_time_utc"] = server_to_utc(pd.to_datetime(t.exit_time)).values
    return t


def donchian_bars():
    low = pd.read_parquet(rd.DATA).rename(columns={"bar_time": "ts"})[["ts", "open", "high", "low", "close"]]
    x = rd.run(low, DON)
    t = pd.DataFrame(dict(decision_time=pd.to_datetime(x.entry_time) + BAR, exit_time=pd.to_datetime(x.exit_time) + BAR,
                          d=np.where(x.side == "long", 1, -1), entry=x.entry_price, exit=x.exit_price, outcome=x.outcome, R=x.R))
    stop = bars.atr.reindex(pd.to_datetime(x.entry_time)).values * DON["atr_stop_multiplier"]
    t["stop_dist"] = stop; t["target_dist"] = stop * DON["reward_risk_ratio"]
    t["stop_price"] = t.entry - t.d * t.stop_dist; t["target_price"] = t.entry + t.d * t.target_dist
    t["usd_at_0.2pct_of_25562"] = t.R * EQUITY * RISK
    return finish(t, "backtest_bars_4y", "Donchian", bars)


def tick_window():
    meta = pq.ParquetFile(ctb.TICKS)
    start = pd.Timestamp(meta.read_row_group(0, columns=["time_msc"]).column(0)[0].as_py()).ceil("15min")
    end = pd.Timestamp(meta.read_row_group(meta.metadata.num_row_groups - 1, columns=["time_msc"]).column(0)[-1].as_py()).floor("15min") - BAR
    f = frame[frame.bar_time < end + BAR].reset_index(drop=True)
    return start, end, f


def donchian_ticks(start, end, f):
    ctb.D_N, ctb.D_STRENGTH, ctb.D_ATR_MULT, ctb.D_RR, ctb.D_MIN_STOP = DON["n_period"], DON["min_trend_strength_pct"], 2.0, DON["reward_risk_ratio"], 8.0
    x = ctb.donchian(f, start, end)
    t = pd.DataFrame(dict(decision_time=pd.to_datetime(x.entry_time), exit_time=pd.to_datetime(x.exit_time),
                          d=np.where(x.direction == "long", 1, -1), entry=x.entry, exit=x.exit, outcome=x.reason,
                          lots=x.lots, usd=x.usd))
    sig = bars.reindex(last_closed_bar(t.decision_time))
    t["stop_dist"] = sig.atr.values * 2.0; t["target_dist"] = t.stop_dist * DON["reward_risk_ratio"]
    t["stop_price"] = sig.close.values - t.d * t.stop_dist; t["target_price"] = sig.close.values + t.d * t.target_dist
    t["R"] = t.usd / (EQUITY * RISK)
    return finish(t, "backtest_ticks_6m", "Donchian", m5)


def slp2_bars():
    x = slp2_simulate(frame)
    t = pd.DataFrame(dict(decision_time=pd.to_datetime(x.entry_time), exit_time=pd.to_datetime(x.exit_time) + BAR,
                          d=np.where(x.direction == "long", 1, -1), entry=x.entry_price, exit=x.exit_price, outcome=x.exit_reason,
                          stop_price=x.stop_price, target_price=x.target_price, stop_dist=x.stop_dist, R=x.r_multiple))
    t["target_dist"] = (t.target_price - t.entry).abs()
    t["usd_at_0.2pct_of_25562"] = t.R * EQUITY * RISK
    return finish(t, "backtest_bars_4y", "SLP2", bars)


def slp2_ticks(start, end, f):
    x = slp2_run_ticks(f, start, end)
    usd = ctb.slp2_dollars(x)
    x = x.reset_index(drop=True)
    t = pd.DataFrame(dict(decision_time=pd.to_datetime(x.entry_time), exit_time=pd.to_datetime(x.exit_time),
                          d=np.where(x.direction == "long", 1, -1), entry=x.entry_price, exit=x.exit_price, outcome=x.exit_reason,
                          stop_price=x.stop_price, target_price=x.target_price, stop_dist=x.stop_dist, R=x.r_multiple))
    t["target_dist"] = (t.target_price - t.entry).abs()
    if len(usd) == len(t):
        t["lots"] = usd.lots.values; t["usd"] = usd.usd.values
    return finish(t, "backtest_ticks_6m", "SLP2", m5)


def live_trades():
    if not hasattr(sys.modules.get("MetaTrader5"), "initialize"):
        sys.modules.pop("MetaTrader5", None)                # reevaluate_donchian installs a stub
    import MetaTrader5 as mt5
    mt5.order_send = None                                   # read-only
    if not mt5.initialize():
        print("MT5 not available -- live_trades skipped"); return pd.DataFrame()
    try:
        deals = mt5.history_deals_get(pd.Timestamp("2026-01-01").to_pydatetime(), (pd.Timestamp.utcnow() + pd.Timedelta(days=1)).tz_localize(None).to_pydatetime())
        df = pd.DataFrame([d._asdict() for d in deals or []])
        df = df[df.symbol == "XAUUSD"]
        orders = {o.position_id: o for o in (mt5.history_orders_get(pd.Timestamp("2026-01-01").to_pydatetime(),
                  (pd.Timestamp.utcnow() + pd.Timedelta(days=1)).tz_localize(None).to_pydatetime()) or []) if o.symbol == "XAUUSD"}
    finally:
        mt5.shutdown()
    reasons = {0: "client", 1: "mobile", 2: "web", 3: "expert(bot)", 4: "stop_loss", 5: "take_profit", 6: "stop_out"}
    bots = {991015: "Donchian", 20260223: "SLP2", 0: "manual"}
    rows = []
    for pid, g in df.groupby("position_id"):
        ins, outs = g[g.entry == 0], g[g.entry.isin([1, 3])]
        if ins.empty or outs.empty:
            continue
        i, o = ins.iloc[0], outs.iloc[-1]
        d = 1 if i.type == 0 else -1
        order = orders.get(pid)
        sl = getattr(order, "sl", 0.) or np.nan; tp = getattr(order, "tp", 0.) or np.nan
        net = g.profit.sum() + g.commission.sum() + g.swap.sum() + g.fee.sum()
        rows.append(dict(position_id=int(pid), bot=bots.get(int(i.magic), f"magic {int(i.magic)}"), side="long" if d == 1 else "short",
            volume=float(i.volume), open_time_server=pd.Timestamp(int(i.time), unit="s"), close_time_server=pd.Timestamp(int(o.time), unit="s"),
            open_price=float(i.price), close_price=float(o.price), initial_sl=sl, initial_tp=tp,
            stop_dist=abs(i.price - sl) if sl == sl else np.nan,
            gross_profit=float(g.profit.sum()), commission=float(g.commission.sum()), swap=float(g.swap.sum()), net_usd=float(net),
            R_vs_initial_stop=float(d * (o.price - i.price) / abs(i.price - sl)) if sl == sl and sl != i.price else np.nan,
            close_reason=reasons.get(int(o.reason), str(o.reason)), opened_by=reasons.get(int(i.reason), str(i.reason))))
    t = pd.DataFrame(rows).sort_values("open_time_server")
    if t.empty:
        return t
    t["open_time_utc"] = server_to_utc(t.open_time_server).values; t["close_time_utc"] = server_to_utc(t.close_time_server).values
    t["hold_hours"] = (t.close_time_server - t.open_time_server).dt.total_seconds() / 3600
    t["win"] = (t.net_usd > 0).astype(int)
    t["source"] = "live_account"
    return t


def save(t, name):
    t.to_csv(os.path.join(HERE, name), index=False, encoding="utf-8-sig", float_format="%.6g")
    print(f"{name}: {len(t)} rows, {t.shape[1]} columns", flush=True)


def main():
    """One bot only: Donchian with the live settings (2x ATR stop, min $8, RR 3), one continuous
    history -- bar engine up to the start of the tick data, real-tick simulation after it."""
    start, end, f = tick_window()
    bars_t, ticks_t = donchian_bars(), donchian_ticks(start, end, f)
    bars_t = bars_t[bars_t.decision_time < start]
    last_exit = pd.to_datetime(bars_t.exit_time).max()
    ticks_t = ticks_t[ticks_t.decision_time >= max(start, last_exit)]      # never two positions at once
    bars_t = bars_t.assign(engine="bars (M15 OHLC)", usd=bars_t["usd_at_0.2pct_of_25562"])
    ticks_t = ticks_t.assign(engine="ticks (real)")
    t = pd.concat([bars_t, ticks_t], ignore_index=True).sort_values("decision_time").reset_index(drop=True)
    # the history columns must follow the one combined sequence
    streak, out = 0, []
    for r in t.R:
        out.append(streak); streak = streak + 1 if r <= 0 else 0
    t["prior_loss_streak"] = out
    t["prev_R"] = t.R.shift()
    t["prev_same_side"] = (t.side == t.side.shift()).astype(int)
    t["hours_since_prev_exit"] = (t.decision_time - pd.to_datetime(t.exit_time).shift()).dt.total_seconds() / 3600
    export(t)


# Only what is known BEFORE the order is sent, then the result (last three columns).
# Left out on purpose: columns that were constant for this bot (the H4 trend is always with
# the trade, the stop is always 2 ATR, price is nearly always on the trade's side of the M15
# EMA200), duplicates (stop/target prices, channel levels) and everything measured after entry.
# name: (dtype, unit, description)
SCHEMA = {
    "trade_id": ("int32", "", "sequence number, time order"),
    "hour_utc": ("int8", "hour 0-23", "hour of the decision, UTC"),
    "weekday": ("int8", "0=Mon..4=Fri", "day of week of the decision"),
    "session": ("int8", "code", "UTC session: 0 = Asia 00-06, 1 = London 07-11, 2 = London-NY overlap 12-15, 3 = New York 16-20, 4 = late 21-23"),
    "is_long": ("int8", "0/1", "trade direction: 1 = long (buy), 0 = short (sell)"),
    "entry_price": ("float64", "USD/oz", "planned entry = close of the signal bar"),
    "m15_atr14_usd": ("float32", "USD/oz", "ATR(14) of M15 bars at the signal bar"),
    "m15_atr_vs_5d_avg": ("float32", "ratio", "M15 ATR divided by its average over the previous 5 days (above 1 = more volatile than usual)"),
    "d1_atr14_usd": ("float32", "USD/oz", "daily ATR(14), previous completed day"),
    "d1_adx14": ("float32", "0-100", "daily ADX(14), previous completed day: trend strength, not direction"),
    "d1_trend_with_trade": ("int8", "0/1", "1 = previous day closed on the trade side of the daily EMA50"),
    "day1_change_pct": ("float32", "%", "gold price change on the 1st last completed trading day before the trade (close vs previous close), positive = price went up"),
    "day2_change_pct": ("float32", "%", "gold price change on the 2nd last completed trading day before the trade (close vs previous close), positive = price went up"),
    "day3_change_pct": ("float32", "%", "gold price change on the 3rd last completed trading day before the trade (close vs previous close), positive = price went up"),
    "day4_change_pct": ("float32", "%", "gold price change on the 4th last completed trading day before the trade (close vs previous close), positive = price went up"),
    "day5_change_pct": ("float32", "%", "gold price change on the 5th last completed trading day before the trade (close vs previous close), positive = price went up"),
    "day6_change_pct": ("float32", "%", "gold price change on the 6th last completed trading day before the trade (close vs previous close), positive = price went up"),
    "day7_change_pct": ("float32", "%", "gold price change on the 7th last completed trading day before the trade (close vs previous close), positive = price went up"),
    "month_change_pct": ("float32", "%", "gold price change over the 30 calendar days before the trade (last completed day close vs the close 30 days earlier), positive = up"),
    "h4c1_change_pct": ("float32", "%", "last completed H4 candle before the trade: close vs open, positive = up candle"),
    "h4c1_range_pct": ("float32", "%", "last completed H4 candle: high minus low, as % of its open (candle size)"),
    "h4c1_close_position": ("float32", "0-1", "last completed H4 candle: where it closed inside its range, 0 = at the low, 1 = at the high"),
    "h4c2_change_pct": ("float32", "%", "2nd last completed H4 candle before the trade: close vs open, positive = up candle"),
    "h4c2_range_pct": ("float32", "%", "2nd last completed H4 candle: high minus low, as % of its open (candle size)"),
    "h4c2_close_position": ("float32", "0-1", "2nd last completed H4 candle: where it closed inside its range, 0 = at the low, 1 = at the high"),
    "h4c3_change_pct": ("float32", "%", "3rd last completed H4 candle before the trade: close vs open, positive = up candle"),
    "h4c3_range_pct": ("float32", "%", "3rd last completed H4 candle: high minus low, as % of its open (candle size)"),
    "h4c3_close_position": ("float32", "0-1", "3rd last completed H4 candle: where it closed inside its range, 0 = at the low, 1 = at the high"),
    "h4c4_change_pct": ("float32", "%", "4th last completed H4 candle before the trade: close vs open, positive = up candle"),
    "h4c4_range_pct": ("float32", "%", "4th last completed H4 candle: high minus low, as % of its open (candle size)"),
    "h4c4_close_position": ("float32", "0-1", "4th last completed H4 candle: where it closed inside its range, 0 = at the low, 1 = at the high"),
    "h4c5_change_pct": ("float32", "%", "5th last completed H4 candle before the trade: close vs open, positive = up candle"),
    "h4c5_range_pct": ("float32", "%", "5th last completed H4 candle: high minus low, as % of its open (candle size)"),
    "h4c5_close_position": ("float32", "0-1", "5th last completed H4 candle: where it closed inside its range, 0 = at the low, 1 = at the high"),
    "h4_trend_age_bars": ("int16", "H4 bars", "H4 bars since price last crossed the H4 EMA30 (6 bars = 1 day)"),
    "h4_ema30_distance_pct": ("float32", "%", "distance of the last closed H4 close from the H4 EMA30, positive = in the trade direction"),
    "h4_ema30_slope_1d_pct": ("float32", "%", "change of the H4 EMA30 over the last day, positive = in the trade direction"),
    "channel_width_atr": ("float32", "ATR", "width of the 20-bar Donchian channel in ATR"),
    "breakout_size_atr": ("float32", "ATR", "how far the signal bar closed beyond the channel, in ATR"),
    "signal_bar_range_atr": ("float32", "ATR", "high minus low of the signal bar, in ATR"),
    "signal_bar_close_position": ("float32", "0-1", "where the signal bar closed inside its own range, 1 = at the extreme in the trade direction"),
    "move_1h_atr": ("float32", "ATR", "price move over the previous hour, positive = in the trade direction"),
    "move_4h_atr": ("float32", "ATR", "price move over the previous 4 hours, positive = in the trade direction"),
    "move_24h_atr": ("float32", "ATR", "price move over the previous 24 hours, positive = in the trade direction"),
    "signal_bar_volume_vs_20": ("float32", "ratio", "tick volume of the signal bar / average of the 20 bars before it (above 1 = busier breakout)"),
    "volume_4h_vs_5d": ("float32", "ratio", "tick volume of the last 4 hours / average 4-hour tick volume of the 5 days before"),
    "spread_vs_5d": ("float32", "ratio", "average spread of the signal bar / its 5-day average (above 1 = wider than usual)"),
    "eurusd_4h_pct": ("float32", "%", "EURUSD change over the last 4 hours (positive = dollar weaker)"),
    "eurusd_24h_pct": ("float32", "%", "EURUSD change over the last 24 hours (positive = dollar weaker)"),
    "eurusd_5d_pct": ("float32", "%", "EURUSD change over the last 5 trading days (positive = dollar weaker)"),
    "usdjpy_24h_pct": ("float32", "%", "USDJPY change over the last 24 hours (positive = dollar stronger)"),
    "silver_24h_pct": ("float32", "%", "XAGUSD change over the last 24 hours"),
    "silver_minus_gold_24h_pct": ("float32", "%", "silver 24h change minus gold 24h change (positive = silver stronger than gold)"),
    "spx500_24h_pct": ("float32", "%", "S&P 500 change over the last 24 hours (history from 2022-10; earlier trades filled with the mode)"),
    "spx500_5d_pct": ("float32", "%", "S&P 500 change over the last 5 trading days"),
    "prior_loss_streak": ("int16", "trades", "consecutive losing trades immediately before this one"),
    "prev_trade_R": ("float32", "R", "result of the previous trade (empty for the first trade)"),
    "prev_trade_same_side": ("int8", "0/1", "1 = the previous trade had the same direction"),
    "hours_since_prev_exit": ("float32", "hours", "time since the previous trade closed (empty for the first trade)"),
    "is_tick_sim": ("int8", "0/1", "how the trade was simulated: 1 = real ticks (from 2026-03-23), 0 = M15 OHLC bars (2022-07 to 2026-03)"),
    "outcome": ("int8", "code", "RESULT, not known before entry: 1 = take profit hit, -1 = stop loss hit, 0 = closed by the 7-day time stop"),
    "win": ("int8", "0/1", "RESULT, not known before entry: 1 = profitable after costs"),
    "R": ("float32", "R", "RESULT, not known before entry: profit in units of risk after costs (1R = 0.2% of the account); a full win is about +3.9R"),
}


INTERMARKET = ("EURUSD", "USDJPY", "XAGUSD", "SPX500")
INTERMARKET_CACHE = os.path.join(HERE, "intermarket_h1.parquet")


def _intermarket_h1():
    """H1 closes of the related symbols on the same broker clock; fetched once (read-only MT5), then cached."""
    if os.path.exists(INTERMARKET_CACHE):
        return pd.read_parquet(INTERMARKET_CACHE)
    stub = sys.modules.pop("MetaTrader5", None)
    try:
        import MetaTrader5 as real_mt5
        real_mt5.order_send = None
        if not real_mt5.initialize():
            raise RuntimeError("MT5 not available")
        cols = {}
        for sym in INTERMARKET:
            real_mt5.symbol_select(sym, True)
            r = real_mt5.copy_rates_range(sym, real_mt5.TIMEFRAME_H1, pd.Timestamp("2022-04-01").to_pydatetime(), pd.Timestamp.now().to_pydatetime())
            cols[sym] = pd.Series(r["close"], index=pd.to_datetime(r["time"], unit="s"))
        real_mt5.shutdown()
    finally:
        if stub is not None and not hasattr(stub, "copy_rates_range"):
            sys.modules["MetaTrader5"] = stub
    df = pd.DataFrame(cols).sort_index()
    df.to_parquet(INTERMARKET_CACHE)
    return df


def intermarket(decision_time):
    """Changes of the related markets up to the last H1 bar that had CLOSED at the decision."""
    im = _intermarket_h1()
    dec = pd.to_datetime(decision_time).values
    out = {}
    gold = bars.close
    def change(series, hours):
        s = series.dropna()
        last = s.index.searchsorted(dec - np.timedelta64(1, "h"), side="right") - 1      # bar open + 1h <= decision
        past = s.index.searchsorted(s.index[np.clip(last, 0, None)] - np.timedelta64(hours, "h"), side="right") - 1
        ok = (last >= 0) & (past >= 0) & (s.index[np.clip(past, 0, None)] >= s.index[0] + np.timedelta64(0, "h"))
        v = (s.values[np.clip(last, 0, None)] / s.values[np.clip(past, 0, None)] - 1) * 100
        return np.where(ok & (past < last), v, np.nan)
    out["eurusd_4h_pct"] = change(im.EURUSD, 4)
    out["eurusd_24h_pct"] = change(im.EURUSD, 24)
    out["eurusd_5d_pct"] = change(im.EURUSD, 24 * 7)
    out["usdjpy_24h_pct"] = change(im.USDJPY, 24)
    out["silver_24h_pct"] = change(im.XAGUSD, 24)
    gold_h1 = gold.resample("1h").last().dropna()
    out["silver_minus_gold_24h_pct"] = out["silver_24h_pct"] - change(gold_h1, 24)
    out["spx500_24h_pct"] = change(im.SPX500, 24)
    out["spx500_5d_pct"] = change(im.SPX500, 24 * 7)
    return out


def activity(decision_time):
    s = bars.reindex(last_closed_bar(decision_time))
    return {"signal_bar_volume_vs_20": s.vol_vs_20.values, "volume_4h_vs_5d": s.vol4h_vs_5d.values, "spread_vs_5d": s.spread_vs_5d.values}


def h4_candles(decision_time, n=5):
    """The last n H4 candles that had CLOSED when the order was sent (1 = most recent).
    Raw market direction, not relative to the trade side."""
    last = h4.known.searchsorted(pd.to_datetime(decision_time).values, side="right") - 1
    o, hi, lo, c = (h4[x].to_numpy(float) for x in ("open", "high", "low", "close"))
    out = {}
    for k in range(1, n + 1):
        i = np.clip(last - (k - 1), 0, None); ok = (last - (k - 1)) >= 0
        rng_ = hi[i] - lo[i]
        out[f"h4c{k}_change_pct"] = np.where(ok, (c[i] / o[i] - 1) * 100, np.nan)
        out[f"h4c{k}_range_pct"] = np.where(ok, rng_ / o[i] * 100, np.nan)
        out[f"h4c{k}_close_position"] = np.where(ok & (rng_ > 0), (c[i] - lo[i]) / np.where(rng_ > 0, rng_, 1), np.nan)
    return out


def daily_moves(decision_time):
    """Market direction per day for the 7 last COMPLETED trading days, and over the last 30 days.
    Raw price direction (positive = up), not relative to the trade side."""
    ret = daily_close.pct_change() * 100
    day = pd.to_datetime(decision_time).dt.normalize().values
    last = daily_close.index.searchsorted(day, side="left") - 1          # yesterday = last day that had closed
    out = {}
    for k in range(1, 8):
        i = last - (k - 1)
        out[f"day{k}_change_pct"] = np.where(i >= 1, ret.values[np.clip(i, 0, None)], np.nan)
    ago = daily_close.index.searchsorted(daily_close.index[last] - pd.Timedelta(days=30), side="right") - 1
    out["month_change_pct"] = np.where(ago >= 0, (daily_close.values[last] / daily_close.values[np.clip(ago, 0, None)] - 1) * 100, np.nan)
    return out


def export(t):
    import json
    t = t.sort_values("decision_time").reset_index(drop=True)
    utc = pd.to_datetime(t.decision_time_utc)
    out = pd.DataFrame({
        "trade_id": range(1, len(t) + 1), "hour_utc": utc.dt.hour, "weekday": utc.dt.weekday,
        "session": pd.cut(utc.dt.hour, [-1, 6, 11, 15, 20, 23], labels=False),
        "is_long": (t.side == "long").astype(int), "entry_price": t.entry,
        "m15_atr14_usd": t.atr14_usd, "m15_atr_vs_5d_avg": t.atr_ratio_5d,
        "d1_atr14_usd": t.d1_atr14_usd, "d1_adx14": t.d1_adx14, "d1_trend_with_trade": t.d1_with_ema50,
        **daily_moves(t.decision_time),
        **h4_candles(t.decision_time),
        "h4_trend_age_bars": t.h4_trend_age_bars, "h4_ema30_distance_pct": t.h4_ema30_dist_pct_dir,
        "h4_ema30_slope_1d_pct": t.h4_ema30_slope_1d_pct_dir,
        "channel_width_atr": t.channel10_width_atr, "breakout_size_atr": t.beyond_channel10_atr,
        "signal_bar_range_atr": t.signal_bar_range_atr, "signal_bar_close_position": t.signal_close_in_bar_dir,
        "move_1h_atr": t.move_1h_dir_atr, "move_4h_atr": t.move_4h_dir_atr, "move_24h_atr": t.move_24h_dir_atr,
        **activity(t.decision_time), **intermarket(t.decision_time),
        "prior_loss_streak": t.prior_loss_streak, "prev_trade_R": t.prev_R, "prev_trade_same_side": t.prev_same_side,
        "hours_since_prev_exit": t.hours_since_prev_exit,
        "is_tick_sim": t.engine.str.startswith("ticks").astype(int),
        "outcome": t.outcome.map({"target": 1, "stop": -1, "time_stop": 0}), "win": t.win, "R": t.R,
    })
    assert list(out.columns) == list(SCHEMA), "columns and SCHEMA out of sync"
    # user request 2026-09-30: fill empty cells with the most frequent value (mode) of that column
    filled = {}
    for c in out.columns[out.isna().any()]:
        mode = out[c].astype("float64").round(4).mode().iloc[0]
        filled[c] = dict(value=float(mode), rows=int(out[c].isna().sum()))
        out[c] = out[c].fillna(mode)
    for c, (dt, _, _) in SCHEMA.items():
        if dt.startswith("int"):
            out[c] = out[c].astype(dt)
        elif dt == "float32":
            out[c] = out[c].astype("float64").round(4).astype("float32")
        elif dt == "category":
            out[c] = out[c].astype("category")
    # every column, kept for re-analysis (feature_selection.py reads this one)
    out.to_parquet(os.path.join(HERE, "donchian_dataset_full.parquet"), index=False)
    # columns judged indistinguishable from chance by feature_selection.py are left out of the main files
    removed = {}
    fs = os.path.join(HERE, "feature_selection.json")
    if os.path.exists(fs):
        sel = json.load(open(fs, encoding="utf-8"))
        why = {r["column"]: r for r in sel["results"]}
        removed = {c: dict(auc_win=why[c]["auc_win"], p_win=why[c]["p_win"], p_R=why[c]["p_R"],
                           same_direction_blocks=why[c]["same_direction_blocks"], reason=why[c]["why"] or "no link with the result")
                   for c in sel["remove"] if c in out.columns}
        out = out.drop(columns=list(removed))
    out.to_parquet(os.path.join(HERE, "donchian_dataset.parquet"), index=False)
    out.to_csv(os.path.join(HERE, "donchian_dataset.csv"), index=False, encoding="utf-8", float_format="%.4f")
    json.dump(dict(
        dataset="Donchian M15 XAUUSD trades: one bot, the live settings",
        settings="10-bar channel breakout on M15, only with the H4 EMA30 trend (price at least 0.5% away), "
                 "stop 2 x ATR14 (= 2 x m15_atr14_usd) with an 8 USD minimum, take profit 3R, risk 0.2% per trade, 2h pause after 3 losses, 7-day time stop",
        costs="spread 0.30 USD/oz, commission 7 USD/lot, FundedNext swap (bars part only)",
        rows=len(out), label_columns=["outcome", "win", "R"],
        leakage="every column before `outcome` is known when the order is sent; outcome, win and R are the result",
        missing="no empty cells: prev_trade_R and hours_since_prev_exit of trade 1 (no previous trade) were filled with the column mode",
        filled_with_mode=filled,
        columns=[dict(name=c, dtype=d, unit=u, description=x) for c, (d, u, x) in SCHEMA.items() if c in out.columns],
        removed_as_chance=removed,
        removed_rule="see feature_selection.py: removed = no permutation-test link (p >= 0.05) with win or R AND not a stable "
                     "direction in >= 3 of 4 time blocks with |AUC - 0.5| >= 0.03, or a duplicate of a stronger column; "
                     "all columns are still in donchian_dataset_full.parquet",
    ), open(os.path.join(HERE, "schema.json"), "w", encoding="utf-8"), indent=1)
    print(f"donchian_dataset: {len(out)} rows, {out.shape[1]} columns (csv + parquet + schema.json)", flush=True)


if __name__ == "__main__":
    main()
