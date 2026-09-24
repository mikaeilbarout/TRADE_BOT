"""Donchian M15 re-entry rule test on an M5 intrabar path (FundedNext bars).

Live behaviour: the bot polls every 3 s; when flat and the LAST CLOSED M15 bar still
signals, it enters -- also seconds after the previous trade closed, on the same bar.
Rules (fixed before running):
  live         -- as above (re-entry at the exit price, same bar)
  next_bar     -- PRIMARY hypothesis: after an exit, no entry until a new M15 bar opens
  fresh_signal -- information: after an exit, the signal must first disappear for a bar
Decision period = 2025-05-01 .. 2026-03-22 (NOT the tick window where the problem was
found). Rule: next_bar is adopted only if there R > live R and DD <= live DD.
The tick window (2026-03-23 ..) is reported separately to calibrate against ticks.
Path model: entries at an M5 open (ask for longs, bid for shorts); stops/targets from
M5 high/low on the exit side, stop first if both, gaps fill at the open. Sizing and
costs as live (0.2% of 25,561.75, lots to 0.01, 7 $/lot); swap: FundedNext per night.
"""
import os, sys, json
import numpy as np
import pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
sys.path.insert(0,ROOT)
from strategy.donchian import add_donchian_indicators, add_trend_indicator, trend_direction, donchian_signal

DATA=os.path.join(ROOT,"..","SLP2","data")
EQUITY,RISK,COMM,CONTRACT=25561.75,.002,7.,100.
SWAP_LONG,SWAP_SHORT=-107.151,-46.917   # points per lot per night (point = 0.01 $/oz)
N,ATRP,EMA,STRENGTH,MULT,RR=10,14,30,.5,3.,3.
TIME_STOP=pd.Timedelta(days=7); M15=pd.Timedelta(minutes=15)
DECISION=(pd.Timestamp("2025-05-01"),pd.Timestamp("2026-03-22 23:59"))
TICKWIN=(pd.Timestamp("2026-03-23 01:15"),pd.Timestamp("2026-09-22 13:00"))


def load():
    m15=pd.read_parquet(os.path.join(DATA,"XAUUSD_M15_5y.parquet"))[["bar_time","open","high","low","close"]].rename(columns={"bar_time":"ts"})
    m5=pd.read_parquet(os.path.join(DATA,"XAUUSD_M5_full_history.parquet"))
    m5["spr"]=m5.spread*0.01
    low=add_donchian_indicators(m15,N,ATRP)
    h4=m15.set_index("ts")[["open","high","low","close"]].resample("4h",label="left",closed="left").agg(
        {"open":"first","high":"max","low":"min","close":"last"}).dropna()
    h4=add_trend_indicator(h4,EMA)
    return low,h4,m5


def simulate(low,h4,m5,rule,start,end):
    h4_close=h4.index.values+np.timedelta64(4,"h")
    sub=m5.set_index("bar_time")
    rows=[]; pos=None; block_until_bar=None; need_gap=False; losses=0; paused=None
    for j in range(1,len(low)):
        t0=low.ts.iloc[j]
        if t0<start: continue
        if t0>=end: break
        row=low.iloc[j-1]
        k=int(np.searchsorted(h4_close,t0.to_datetime64(),side="right"))-1
        trend=trend_direction(h4.iloc[k],STRENGTH) if k>=0 else "flat"
        sig=donchian_signal(row,trend) if trend!="flat" else None
        if rule=="fresh_signal" and need_gap and sig is None:
            need_gap=False
        bars=sub.loc[t0:t0+M15-pd.Timedelta(seconds=1)]
        for ts,b in bars.iterrows():
            reenter_at=None
            if pos is not None:
                d=pos["d"]; spr=b.spr if d==-1 else 0.
                o,h,l=b.open+spr,b.high+spr,b.low+spr
                reason=px=None
                if d*(o-pos["sl"])<=0: reason,px="stop",o
                elif d*(o-pos["tp"])>=0: reason,px="target",pos["tp"]
                elif ts>=pos["deadline"]: reason,px="time_stop",o
                elif (l<=pos["sl"] if d==1 else h>=pos["sl"]): reason,px="stop",pos["sl"]
                elif (h>=pos["tp"] if d==1 else l<=pos["tp"]): reason,px="target",pos["tp"]
                if reason:
                    nights=(ts.normalize()-pos["t"].normalize()).days
                    swap=pos["lots"]*(SWAP_LONG if d==1 else SWAP_SHORT)*nights  # points/lot = $/lot (0.01 $/oz x 100 oz)
                    usd=pos["lots"]*(CONTRACT*d*(px-pos["entry"])-COMM)+swap
                    rows.append(dict(entry_time=pos["t"],exit_time=ts,direction="long" if d==1 else "short",
                                     reason=reason,usd=usd,same_bar_reentry=pos["reentry"]))
                    losses=0 if usd>0 else losses+1
                    if losses>=3: paused=ts+pd.Timedelta(hours=2); losses=0
                    pos=None
                    if rule=="next_bar": block_until_bar=t0+M15
                    if rule=="fresh_signal": need_gap=True
                    reenter_at=px if reason in ("target","stop") else None
                else:
                    continue
            if pos is None and sig is not None and not (paused is not None and ts<paused):
                if block_until_bar is not None and t0<block_until_bar: continue
                if rule=="fresh_signal" and need_gap: continue
                d=1 if sig=="long" else -1
                stop_dist=row.atr*MULT; sl=row.close-d*stop_dist; tp=row.close+d*stop_dist*RR
                bid=reenter_at if reenter_at is not None else b.open
                if reenter_at is not None and d==-1: bid=reenter_at-b.spr   # exit fill was an ask
                entry=bid+(b.spr if d==1 else 0.)
                if d*((entry-(b.spr if d==1 else 0.))-sl)<=0 or d*(tp-entry)<=0: continue
                lots=max(.01,round(round(EQUITY*RISK/stop_dist/CONTRACT/.01)*.01,2))
                pos=dict(d=d,entry=entry,sl=sl,tp=tp,lots=lots,t=ts,deadline=ts+TIME_STOP,reentry=reenter_at is not None)
    return pd.DataFrame(rows)


def stats(t):
    if t.empty: return dict(n=0)
    r=t.sort_values("exit_time").usd/(EQUITY*RISK); c=r.cumsum()
    return dict(n=len(r),wins=int((r>0).sum()),win_rate=round(float((r>0).mean()),3),R=round(float(r.sum()),1),
        usd=round(float(t.usd.sum()),0),PF=round(float(r.clip(lower=0).sum()/-r.clip(upper=0).sum()),2),
        DD_R=round(float((c.cummax().clip(lower=0)-c).max()),1),same_bar_reentries=int(t.same_bar_reentry.sum()),
        reentry_R=round(float(t[t.same_bar_reentry].usd.sum()/(EQUITY*RISK)),1))


def main():
    low,h4,m5=load()
    out=dict(decision_period=[str(x) for x in DECISION],tick_window=[str(x) for x in TICKWIN],results={})
    for name,(a,b) in (("decision",DECISION),("tick_window",TICKWIN)):
        for rule in ("live","next_bar","fresh_signal"):
            t=simulate(low,h4,m5,rule,a,b)
            t.to_csv(os.path.join(HERE,f"reentry_{name}_{rule}.csv"),index=False)
            out["results"][f"{name}/{rule}"]=stats(t); print(name,rule,out["results"][f"{name}/{rule}"],flush=True)
    d=out["results"]
    out["adopt_next_bar"]=bool(d["decision/next_bar"]["R"]>d["decision/live"]["R"] and d["decision/next_bar"]["DD_R"]<=d["decision/live"]["DD_R"])
    with open(os.path.join(HERE,"reentry_report.json"),"w") as f: json.dump(out,f,indent=1)
    print("adopt next_bar:",out["adopt_next_bar"])


if __name__=="__main__":
    main()
