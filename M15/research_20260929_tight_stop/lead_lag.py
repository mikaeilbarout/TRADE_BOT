"""Does the dollar move BEFORE gold on short time scales? (user question 2026-09-30)

M1 closes from the broker (read-only MT5) for XAUUSD and 7 dollar pairs, last ~2 months (MT5 allows about 60 days of M1 here).
Dollar index = equal-weight USD side of the pairs' log changes (as in dollar_index_filter.py).
1. Cross-correlation corr(gold return at t, dollar return at t - lag) for lags -10..+10 minutes:
   a peak at a POSITIVE lag would mean the dollar leads gold.
2. Predictive check: does the dollar's move over the last 1/3/5 minutes predict gold's move over the
   NEXT 1/5/15 minutes? (rank correlation, first half vs second half of the sample)
"""
import sys, os, json
import numpy as np
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
PAIRS = {"EURUSD": -1, "GBPUSD": -1, "AUDUSD": -1, "NZDUSD": -1, "USDJPY": 1, "USDCHF": 1, "USDCAD": 1}
CACHE = os.path.join(HERE, "m1_gold_usd.parquet")


def load():
    if os.path.exists(CACHE):
        return pd.read_parquet(CACHE)
    import MetaTrader5 as mt5, time
    mt5.order_send = None
    assert mt5.initialize(), mt5.last_error()
    cols = {}
    for s in ["XAUUSD"] + list(PAIRS):
        mt5.symbol_select(s, True)
        for _ in range(6):
            r = mt5.copy_rates_range(s, mt5.TIMEFRAME_M1, (pd.Timestamp.now() - pd.Timedelta(days=60)).to_pydatetime(), pd.Timestamp.now().to_pydatetime())
            if r is not None and len(r) > 10000:
                break
            time.sleep(5)
        if r is None or len(r) == 0:
            print(s, "no M1 data:", mt5.last_error()); continue
        cols[s] = pd.Series(r["close"], index=pd.to_datetime(r["time"], unit="s"))
        print(s, len(r), cols[s].index[0], cols[s].index[-1], flush=True)
    mt5.shutdown()
    df = pd.DataFrame(cols).sort_index()
    df.to_parquet(CACHE)
    return df


def main():
    df = load()
    start = max(df[c].first_valid_index() for c in df.columns)
    df = df[df.index >= start].dropna()                       # minutes where every symbol has a bar
    lg = np.log(df)
    ret = lg.diff()
    gap = df.index.to_series().diff() != pd.Timedelta(minutes=1)
    ret[gap.values] = np.nan                                   # no returns across market gaps
    gold = ret.XAUUSD
    pairs = {s: v for s, v in PAIRS.items() if s in ret.columns}
    print("dollar index from:", list(pairs))
    usd = sum(ret[s] * sign for s, sign in pairs.items()) / len(pairs)
    print(f"\n{len(df)} common M1 bars, {df.index[0]} .. {df.index[-1]}")
    cc = {}
    for lag in range(-10, 11):
        cc[lag] = round(float(gold.corr(usd.shift(lag))), 4)
    print("\ncorr(gold at t, dollar at t-lag)   [lag>0: dollar earlier]")
    for lag, v in cc.items():
        print(f"  lag {lag:+3d} min: {v:+.4f} {'#' * int(abs(v) * 100)}")
    out = dict(bars=len(df), period=[str(df.index[0]), str(df.index[-1])], cross_corr=cc, predictive={})
    half = len(df) // 2
    print("\npredictive: dollar move over the last N min vs gold move over the NEXT M min (rank corr; 1st half / 2nd half)")
    for back in (1, 3, 5):
        u_past = usd.rolling(back).sum()
        for fwd in (1, 5, 15):
            g_next = gold[::-1].rolling(fwd).sum()[::-1].shift(-1)
            d = pd.concat([u_past, g_next], axis=1).dropna()
            r1 = d.iloc[:half].rank().corr().iloc[0, 1]; r2 = d.iloc[half:].rank().corr().iloc[0, 1]
            out["predictive"][f"usd_last_{back}m->gold_next_{fwd}m"] = [round(float(r1), 4), round(float(r2), 4)]
            print(f"  dollar last {back} min -> gold next {fwd:2d} min: {r1:+.4f} / {r2:+.4f}")
    json.dump(out, open(os.path.join(HERE, "lead_lag.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
