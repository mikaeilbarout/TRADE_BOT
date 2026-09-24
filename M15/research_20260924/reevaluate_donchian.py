"""Re-evaluate the Donchian M15 profile with the look-ahead-free engine (2026-09-24).

Data: FundedNext M15 bars (the live broker, server time; H4 resampled from them).
Costs: spread 0.30 $/oz, commission 7 $/lot, FundedNext swap read live 2026-09-24
(long -107.151, short -46.917 points/lot/night = -1.0715 / -0.4692 $/oz/night; the
engine ignores triple-swap Wednesday, so real swap is somewhat higher). Risk 0.2%,
3-loss/2h cooldown, 7-day time stop -- as live.
Protocol (saved before any run): full grid below; choose on the FIRST 70% only by the
neighbourhood-averaged train score (cell and its +-1 grid neighbours), then one test of
the winner vs the current profile on the last 30%. Output: this folder.
"""
import sys, os, json, itertools, types, importlib
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
sys.path.insert(0,ROOT)
mt5=types.ModuleType("MetaTrader5")
for k in ("TIMEFRAME_M1","TIMEFRAME_M5","TIMEFRAME_M15","TIMEFRAME_M30","TIMEFRAME_H1","TIMEFRAME_H4","TIMEFRAME_D1"): setattr(mt5,k,1)
sys.modules.setdefault("MetaTrader5",mt5)
import numpy as np
import pandas as pd
from strategy.donchian import simulate_donchian

PROFILE=importlib.import_module("mt5.profiles.profile_m15")
DATA=os.path.join(ROOT,"..","SLP2","data","XAUUSD_M15_5y.parquet")
COSTS=dict(spread_dollars=.30,commission_dollars=.07,swap_long=-1.07151,swap_short=-.46917)
STRESS=dict(spread_dollars=.50,commission_dollars=.10,swap_long=-1.07151*1.4,swap_short=-.46917*1.4)  # +40% swap ~ triple-Wednesday
GRID=dict(n_period=[10,20,40],ema_trend_period=[30,50,100],min_trend_strength_pct=[0.,.3,.5],
          atr_stop_multiplier=[2.,3.,4.],reward_risk_ratio=[1.5,2.,3.])
CURRENT=dict(n_period=10,ema_trend_period=30,min_trend_strength_pct=.5,atr_stop_multiplier=3.,reward_risk_ratio=3.)
RULE=("winner replaces the current profile only if last-30% trades>=40, PF>1, total_R>current, "
      "DD_R<=1.5*current and stress PF>1; no fallback to the runner-up")


def frames(low):
    high=low.set_index("ts").resample("4h",label="left",closed="left").agg(
        {"open":"first","high":"max","low":"min","close":"last"}).dropna().reset_index()
    return low,high


def run(low,p,costs=COSTS,start=None):
    lo,hi=frames(low)
    t,_=simulate_donchian(lo,hi,atr_period=14,time_stop_minutes=10080,risk_cfg=PROFILE.RISK,starting_equity=10000.,
        cooldown_losses_to_trigger=3,cooldown_hours=2.,**p,**costs)
    if t.empty:
        return t
    t["R"]=t.pnl/((t.equity_after-t.pnl)*PROFILE.RISK.risk_per_trade_pct/100)
    if start is not None:
        t=t[pd.to_datetime(t.entry_time)>=start]
    return t.reset_index(drop=True)


def summary(t):
    if len(t)==0:
        return dict(n=0,win=None,R=0.,PF=None,DD=0.)
    r=t.R.astype(float); c=r.cumsum(); dd=float((c.cummax().clip(lower=0)-c).max())
    loss=-r.clip(upper=0).sum()
    return dict(n=len(r),win=float((r>0).mean()),R=float(r.sum()),PF=float(r.clip(lower=0).sum()/loss) if loss>0 else None,DD=dd)


def score(train,t):
    thirds=[train.ts.iloc[int(len(train)*j/3)] for j in (1,2)]
    e=pd.to_datetime(t.entry_time)
    blocks=[summary(t[m]) for m in (e<thirds[0],(e>=thirds[0])&(e<thirds[1]),e>=thirds[1])]
    if not all(b["n"]>=10 for b in blocks):
        return None
    s=summary(t)
    return s["R"]/max(5.,s["DD"])-.5*sum(b["R"]<0 for b in blocks)


def main():
    with open(os.path.join(HERE,"protocol.json"),"w") as f:
        json.dump(dict(grid=GRID,current=CURRENT,costs=COSTS,stress=STRESS,rule=RULE,
            selection="first 70%: total_R/max(5,DD_R)-0.5 per losing third, averaged over cell and +-1 neighbours"),f,indent=1)
    low=pd.read_parquet(DATA).rename(columns={"bar_time":"ts"})[["ts","open","high","low","close"]]
    split=int(len(low)*.7); boundary=low.ts.iloc[split]
    train=low.iloc[:split].reset_index(drop=True)
    test=low.iloc[split-6000:].reset_index(drop=True)   # ~2 months warm-up for EMA/ATR
    keys=list(GRID); cells={}
    for combo in itertools.product(*GRID.values()):
        p=dict(zip(keys,combo)); t=run(train,p)
        cells[combo]=dict(p=p,raw=score(train,t),train=summary(t))
    for combo,c in cells.items():
        idx=[GRID[k].index(v) for k,v in zip(keys,combo)]; group=[c]
        for j,k in enumerate(keys):
            for step in (-1,1):
                m=idx[j]+step
                if 0<=m<len(GRID[k]):
                    group.append(cells[tuple(GRID[kk][idx[jj] if jj!=j else m] for jj,kk in enumerate(keys))])
        c["smoothed"]=float(np.mean([g["raw"] if g["raw"] is not None else -1. for g in group])) if c["raw"] is not None else None
    ranked=sorted((c for c in cells.values() if c["smoothed"] is not None),key=lambda c:c["smoothed"],reverse=True)
    winner=ranked[0]["p"]
    res={}
    for label,p in (("current",CURRENT),("selected",winner)):
        res[label]=dict(params=p,train=cells[tuple(p[k] for k in keys)]["train"],
            test=summary(run(test,p,start=boundary)),stress_test=summary(run(test,p,STRESS,start=boundary)),
            full=summary(run(low,p)))
    a,b=res["selected"]["test"],res["current"]["test"]
    promote=bool(winner!=CURRENT and a["n"]>=40 and (a["PF"] or 0)>1 and a["R"]>b["R"] and a["DD"]<=1.5*b["DD"]
                 and (res["selected"]["stress_test"]["PF"] or 0)>1)
    share_pos=float(np.mean([c["train"]["R"]>0 for c in cells.values()]))
    out=dict(data=[str(low.ts.iloc[0]),str(low.ts.iloc[-1])],split_time=str(boundary),results=res,promotion_passed=promote,
             top10_train=[dict(p=c["p"],smoothed=c["smoothed"],raw=c["raw"],train=c["train"]) for c in ranked[:10]],
             share_of_grid_profitable_on_train=share_pos)
    with open(os.path.join(HERE,"report.json"),"w") as f:
        json.dump(out,f,indent=1,default=str)
    print(json.dumps(out,indent=1,default=str))


if __name__=="__main__":
    main()
