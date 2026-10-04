"""Fetch M15 history (read-only MT5) for candidate symbols + their costs, cache to parquet."""
import MetaTrader5 as mt5, pandas as pd, time, os, json
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "multi_m15"); os.makedirs(OUT, exist_ok=True)
SYMS = ["XAUUSD", "XAGUSD", "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD", "SPX500", "NDX100", "US30", "BTCUSD", "XPTUSD", "USOIL", "UKOIL", "XTIUSD", "WTI"]
mt5.order_send = None
assert mt5.initialize(), mt5.last_error()
names = {s.name for s in mt5.symbols_get()}
info = {}
for s in SYMS:
    if s not in names:
        print(s, "not offered"); continue
    mt5.symbol_select(s, True); si = mt5.symbol_info(s)
    r = None
    for _ in range(2):
        r = mt5.copy_rates_from_pos(s, mt5.TIMEFRAME_M15, 0, 90000)
        if r is not None and len(r) > 20000: break
        time.sleep(3)
    if r is None or len(r) < 20000:
        print(s, "no usable history", None if r is None else len(r)); continue
    df = pd.DataFrame(r); df["bar_time"] = pd.to_datetime(df.time, unit="s")
    df[["bar_time", "open", "high", "low", "close", "tick_volume", "spread"]].to_parquet(os.path.join(OUT, f"{s}.parquet"))
    info[s] = dict(bars=len(df), first=str(df.bar_time.iloc[0]), last=str(df.bar_time.iloc[-1]), contract=si.trade_contract_size, point=si.point,
                   median_spread_points=float(df.spread.median()), digits=si.digits, price=float(df.close.iloc[-1]))
    print(s, info[s], flush=True)
json.dump(info, open(os.path.join(OUT, "info.json"), "w"), indent=1)
mt5.shutdown()
