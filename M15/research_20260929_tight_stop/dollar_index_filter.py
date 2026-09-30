"""Broad US-dollar strength vs gold, and a filter built on it (user idea, 2026-09-30):
"if all the dollar pairs move in the dollar's favour, the dollar is strong and gold should fall".

Dollar index = equal-weight average of the USD side of each pair's log change
(XXXUSD falling = dollar up; USDXXX rising = dollar up), from H1 closes on the broker's clock.
Breadth = how many pairs moved in the dollar's favour.

Part 1 -- how strongly gold moves against this index (H1 / H4 / daily correlation).
Part 2 -- the filter, on top of the CURRENT live bot (N20, EMA30, 0.3%, 2 ATR min $8, RR 4 + S&P filter):
  D1 "broad":    skip a gold LONG when, over the last 24 h, ALL pairs (of those with history) moved in the dollar's favour
                 (and a SHORT when ALL moved against the dollar)            -- the user's rule, no parameter
  D2 "strength": skip when the 24 h dollar-index move against the trade is at or above the 67th
                 percentile of that value on the FIRST 70% of the trades    -- one threshold, fixed from data
Adopted only if, vs the live bot, it improves the LAST 30% of the 4 years (R, DD <= 1.2x)
AND the 6-month real ticks (net USD). Two variants tested -> one passing by luck is possible.
"""
import sys, os, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import pandas as pd
import counter_move_filters as cm
import spx_filter as spx
rd = cm.rd

PAIRS = {"EURUSD": -1, "GBPUSD": -1, "AUDUSD": -1, "NZDUSD": -1, "USDJPY": 1, "USDCHF": 1, "USDCAD": 1}   # +1 = price up means dollar up
CACHE = os.path.join(cm.COMB, "trade_dataset_20260929", "usd_pairs_h1.parquet")
LIVE = dict(n_period=20, ema_trend_period=30, min_trend_strength_pct=0.3, atr_stop_multiplier=2.0, reward_risk_ratio=4.0, min_stop_dollars=8.0)
BAR = pd.Timedelta(minutes=15)


def load_pairs():
    if os.path.exists(CACHE):
        return pd.read_parquet(CACHE)
    sys.modules.pop("MetaTrader5", None)
    import MetaTrader5 as mt5
    mt5.order_send = None                              # read-only
    assert mt5.initialize(), mt5.last_error()
    cols = {}
    for s in PAIRS:
        mt5.symbol_select(s, True)
        r = None
        for _ in range(5):                               # history may still be downloading
            r = mt5.copy_rates_range(s, mt5.TIMEFRAME_H1, pd.Timestamp("2022-04-01").to_pydatetime(), pd.Timestamp.now().to_pydatetime())
            if r is not None and len(r) > 1000:
                break
            import time; time.sleep(5)
        if r is None or len(r) <= 1000:
            print(f"{s}: no usable history on this broker -- left out")
            continue
        cols[s] = pd.Series(r["close"], index=pd.to_datetime(r["time"], unit="s"))
    mt5.shutdown()
    df = pd.DataFrame(cols).sort_index()
    df.to_parquet(CACHE)
    return df


PX = load_pairs().ffill()
PAIRS = {s: sign for s, sign in PAIRS.items() if s in PX.columns}   # only pairs with history
print("dollar index built from:", list(PAIRS))
# per-pair "dollar up" log change and the equal-weight index level (log, x100 = %)
USD_LOG = sum(np.log(PX[s]) * sign for s, sign in PAIRS.items()) / len(PAIRS) * 100


def usd_state(signal_bar_ts, hours=24):
    """(index move %, pairs moving for the dollar) over `hours` up to the last CLOSED H1 bar."""
    dec = pd.Timestamp(signal_bar_ts) + BAR
    idx = PX.index
    last = idx.searchsorted(dec - pd.Timedelta(hours=1), side="right") - 1
    if last < 0:
        return np.nan, np.nan
    past = idx.searchsorted(idx[last] - pd.Timedelta(hours=hours), side="right") - 1
    if past < 0 or past >= last:
        return np.nan, np.nan
    move = USD_LOG.iloc[last] - USD_LOG.iloc[past]
    ups = sum(int(np.sign(PX[s].iloc[last] - PX[s].iloc[past]) == sign) for s, sign in PAIRS.items())
    return float(move), int(ups)


def against_trade(move, side):
    """dollar move AGAINST the gold trade: dollar up hurts a long, dollar down hurts a short"""
    return move if side == "long" else -move


def make_d1():
    def f(ts, side):
        move, ups = usd_state(ts)
        if np.isnan(move):
            return True
        return not ((side == "long" and ups == len(PAIRS)) or (side == "short" and ups == 0))
    return f


def make_d2(threshold):
    def f(ts, side):
        move, _ = usd_state(ts)
        return True if np.isnan(move) else against_trade(move, side) < threshold
    return f


def both(*fs):
    return lambda ts, side: all(f(ts, side) for f in fs)


def part1():
    gold = pd.read_parquet(rd.DATA).set_index("bar_time").close
    out = {}
    for rule, lab in (("1h", "H1"), ("4h", "H4"), ("1D", "daily")):
        g = np.log(gold.resample(rule).last().dropna()).diff()
        u = USD_LOG.resample(rule).last().diff()
        j = pd.concat([g, u], axis=1, keys=["gold", "usd"]).dropna()
        out[lab] = round(float(j.gold.corr(j.usd)), 3)
    g = np.log(gold.resample("1D").last().dropna()).diff() * 100
    u = USD_LOG.resample("1D").last().diff()
    j = pd.concat([g, u], axis=1, keys=["gold", "usd"]).dropna()
    ups = pd.DataFrame({s: np.sign(PX[s].resample("1D").last().diff()) == sign for s, sign in PAIRS.items()}).sum(axis=1)
    j["ups"] = ups.reindex(j.index)
    by_breadth = j.groupby("ups").gold.agg(days="size", mean_gold_pct="mean", gold_down_share=lambda x: (x < 0).mean()).round(3)
    print("correlation of gold returns with the dollar index:", out)
    print("gold on days by how many of the 7 pairs moved for the dollar:\n" + by_breadth.to_string())
    return dict(correlation=out, by_breadth=by_breadth.reset_index().to_dict("records"))


def main():
    rep = dict(part1=part1())
    low = pd.read_parquet(rd.DATA).rename(columns={"bar_time": "ts"})[["ts", "open", "high", "low", "close"]]
    split = int(len(low) * .7); boundary = low.ts.iloc[split]
    test = low.iloc[split - 6000:].reset_index(drop=True)
    # threshold for D2 from the live bot's trades in the FIRST 70% only
    base_trades = rd.run(low, dict(LIVE, entry_filter=spx.spx_filter))
    base_trades = base_trades[pd.to_datetime(base_trades.entry_time) < boundary]
    vals = [against_trade(usd_state(ts)[0], s) for ts, s in zip(pd.to_datetime(base_trades.entry_time), base_trades.side)]
    thr = float(np.nanpercentile(vals, 100 * 2 / 3))
    print(f"\nD2 threshold (67th pct of first-70% trades): dollar {thr:+.3f}% against the trade over 24 h")

    sys.path.insert(0, cm.COMB); sys.path.insert(0, os.path.join(cm.COMB, "SLP2"))
    import combined_tick_backtest as ctb
    import pyarrow.parquet as pq
    meta = pq.ParquetFile(ctb.TICKS)
    start = pd.Timestamp(meta.read_row_group(0, columns=["time_msc"]).column(0)[0].as_py()).ceil("15min")
    end = pd.Timestamp(meta.read_row_group(meta.metadata.num_row_groups - 1, columns=["time_msc"]).column(0)[-1].as_py()).floor("15min") - ctb.BAR
    frame = ctb.load_m15(); frame = frame[frame.bar_time < end + ctb.BAR].reset_index(drop=True)
    ctb.D_N, ctb.D_EMA, ctb.D_STRENGTH, ctb.D_ATR_MULT, ctb.D_RR, ctb.D_MIN_STOP = 20, 30, 0.3, 2.0, 4.0, 8.0
    variants = {"live (S&P filter)": spx.spx_filter,
                "live + D1 broad (all 7 pairs)": both(spx.spx_filter, make_d1()),
                "live + D2 strength": both(spx.spx_filter, make_d2(thr))}
    res = {}
    for name, filt in variants.items():
        full = rd.run(low, dict(LIVE, entry_filter=filt)); e = pd.to_datetime(full.entry_time)
        last30 = rd.run(test, dict(LIVE, entry_filter=filt), start=boundary)
        ctb.D_ENTRY_FILTER = filt
        tk = ctb.stats(ctb.donchian(frame, start, end))
        res[name] = dict(first70=rd.summary(full[e < boundary]), last30=rd.summary(last30), full=rd.summary(full), tick=tk,
                         years={int(k): round(float(v), 1) for k, v in full.R.groupby(pd.to_datetime(full.exit_time).dt.year).sum().items()})
        a, b, f = res[name]["first70"], res[name]["last30"], res[name]["full"]
        print(f"{name:30} | first70 n={a['n']} win={a['win']:.2f} R={a['R']:6.1f} DD={a['DD']:5.1f} | last30 n={b['n']} win={b['win']:.2f} R={b['R']:6.1f} DD={b['DD']:5.1f} | "
              f"4y R={f['R']:6.1f} DD={f['DD']:5.1f} | TICK n={tk['trades']} win={tk['win_rate']} ${tk['net_usd']} DD {tk['max_dd_pct']}% | {res[name]['years']}", flush=True)
    base = res["live (S&P filter)"]
    for name in list(variants)[1:]:
        r = res[name]
        r["adopt"] = bool(r["last30"]["R"] > base["last30"]["R"] and r["last30"]["DD"] <= 1.2 * base["last30"]["DD"] and r["tick"]["net_usd"] > base["tick"]["net_usd"])
        print(f"ADOPT {name}: {r['adopt']}")
    rep.update(threshold_D2=thr, split=str(boundary), results=res)
    json.dump(rep, open(os.path.join(HERE, "dollar_index_filter.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
