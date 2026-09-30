"""Anatomy of the losing trades: Donchian M15 live settings (2x ATR stop, min stop $8, RR 3)
on 6 months of ticks. For every trade: how far it went in our favour before the exit
(MFE, from M5 bars), what the market did after a stop, and the context at the signal.
Descriptive only -- nothing here is used to change the bot.
"""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); COMB = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, COMB); sys.path.insert(0, os.path.join(COMB, "SLP2")); sys.path.insert(0, os.path.join(COMB, "M15"))
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import combined_tick_backtest as ctb
from strategy.donchian import add_donchian_indicators, add_trend_indicator

pd.set_option("display.width", 200)
meta = pq.ParquetFile(ctb.TICKS)
start = pd.Timestamp(meta.read_row_group(0, columns=["time_msc"]).column(0)[0].as_py()).ceil("15min")
end = pd.Timestamp(meta.read_row_group(meta.metadata.num_row_groups - 1, columns=["time_msc"]).column(0)[-1].as_py()).floor("15min") - ctb.BAR
frame = ctb.load_m15(); frame = frame[frame.bar_time < end + ctb.BAR].reset_index(drop=True)
ctb.D_ATR_MULT, ctb.D_RR, ctb.D_MIN_STOP = 2.0, 3.0, 8.0
t = ctb.donchian(frame, start, end).sort_values("entry_time").reset_index(drop=True)

low = add_donchian_indicators(frame.rename(columns={"bar_time": "ts"})[["ts", "open", "high", "low", "close"]], 10, 14)
h4 = frame.set_index("bar_time")[["open", "high", "low", "close"]].resample("4h", label="left", closed="left").agg(
    {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
h4 = add_trend_indicator(h4, 30)
h4["side"] = np.sign(h4.close - h4.ema_trend)
h4["age"] = h4.side.groupby((h4.side != h4.side.shift()).cumsum()).cumcount() + 1   # H4 bars since the EMA side flipped
h4_close = h4.index + pd.Timedelta(hours=4)
d1 = frame.set_index("bar_time").resample("1D").agg({"high": "max", "low": "min", "close": "last"}).dropna()
d1["ema50"] = d1.close.ewm(span=50, adjust=False).mean()
m5 = pd.read_parquet(os.path.join(COMB, "SLP2", "data", "XAUUSD_M5_full_history.parquet")).set_index("bar_time")

rows = []
for _, x in t.iterrows():
    d = 1 if x.direction == "long" else -1
    j = int(low.ts.searchsorted(x.entry_time.floor("15min"))); sig = low.iloc[j - 1]
    stop_dist = sig.atr * 2.0
    k = int(np.searchsorted(h4_close.values, x.entry_time.floor("15min").to_datetime64(), side="right")) - 1
    hb = h4.iloc[k]
    dd = d1[d1.index < x.entry_time.normalize()].iloc[-1]
    path = m5.loc[x.entry_time.floor("5min"):x.exit_time]
    fav = ((path.high - x.entry) if d == 1 else (x.entry - path.low)).max()
    after = m5.loc[x.exit_time:x.exit_time + pd.Timedelta(hours=24)]
    later_fav = ((after.high - x.entry) if d == 1 else (x.entry - after.low)).max() if len(after) else np.nan
    rows.append(dict(
        entry_time=x.entry_time, side=x.direction, reason=x.reason, usd=x.usd, R=x.usd / (ctb.EQUITY * ctb.RISK_PCT),
        hold_h=(x.exit_time - x.entry_time).total_seconds() / 3600, stop=stop_dist,
        mfe_R=fav / stop_dist,                                                 # best open profit before the exit
        breakout_atr=d * (sig.close - (sig.donchian_high if d == 1 else sig.donchian_low)) / sig.atr,
        channel_atr=(sig.donchian_high - sig.donchian_low) / sig.atr,          # how wide the 10-bar range was
        ema_dist_pct=d * (hb.close - hb.ema_trend) / hb.close * 100,          # H4 trend strength at entry
        h4_trend_age=hb.age,
        with_d1=np.sign(dd.close - dd.ema50) == d,                            # also with the daily EMA50 trend?
        hour=x.entry_time.hour,
        after_stop_went_to_target=bool(x.reason == "stop" and later_fav >= 3 * stop_dist),
    ))
a = pd.DataFrame(rows)
a["prev_loss"] = (a.R.shift() < 0) & (a.side == a.side.shift())
L, W = a[a.R < 0], a[a.R > 0]
print(f"trades {len(a)} | wins {len(W)} | losses {len(L)} | loss total {L.R.sum():.1f}R | win total {W.R.sum():.1f}R\n")

print("=== 1. how the losers died (best open profit before the stop)")
bins = [-1, .25, .5, 1, 2, 99]
labels = ["never beyond +0.25R", "+0.25 to 0.5R", "+0.5 to 1R", "+1 to 2R", "over +2R then stopped"]
c = pd.cut(L.mfe_R, bins, labels=labels).value_counts().reindex(labels)
for lab, n in c.items():
    print(f"  {lab:24} {n:3d} ({n / len(L) * 100:4.1f}%)")
print(f"  median time to stop {L.hold_h.median():.1f}h | stopped within 1h {(L.hold_h <= 1).mean() * 100:.0f}% | within 4h {(L.hold_h <= 4).mean() * 100:.0f}%")
print(f"  after the stop, price went on to the original target within 24h: {L.after_stop_went_to_target.mean() * 100:.0f}% of losers\n")

print("=== 2. winners vs losers at the signal (medians)")
cols = ["breakout_atr", "channel_atr", "ema_dist_pct", "h4_trend_age", "stop"]
print(pd.DataFrame({"winners": W[cols].median(), "losers": L[cols].median()}).round(2).to_string(), "\n")


def table(col, bins=None, labels=None):
    g = a.assign(b=pd.cut(a[col], bins, labels=labels) if bins is not None else a[col]).groupby("b", observed=True)
    out = g.agg(n=("R", "size"), win=("R", lambda r: (r > 0).mean()), R=("R", "sum")).round(2)
    print(out.to_string(), "\n")


print("=== 3. by side"); table("side")
print("=== 4. with / against the daily EMA50 trend"); table("with_d1")
print("=== 5. by breakout size (close beyond the channel, in ATR)"); table("breakout_atr", [-1, .1, .25, .5, 1, 99])
print("=== 6. by H4 trend strength (distance from EMA30, %)"); table("ema_dist_pct", [0, .75, 1, 1.5, 2.5, 99])
print("=== 7. by H4 trend age (H4 bars since it flipped)"); table("h4_trend_age", [0, 3, 6, 12, 24, 999])
print("=== 8. by session (UTC hour of entry)"); table("hour", [-1, 6, 12, 16, 20, 24], ["Asia 0-6", "London 7-12", "NY overlap 13-16", "NY late 17-20", "close 21-23"])
print("=== 9. right after a same-direction loss"); table("prev_loss")
a.to_csv(os.path.join(HERE, "loss_anatomy_trades.csv"), index=False)
