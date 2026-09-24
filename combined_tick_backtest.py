"""Both live bots together on the 6-month tick file, in account dollars.

SLP2   : SLP2/scripts/sp2l_tick_backtest.run_ticks with the live settings (RR=5, no filters),
         sized like SLP2 live (floor lots so stop loss + commission <= 0.2% equity; the
         20-point deviation is added to the stop distance, as in place_order).
Donchian: tick re-implementation of M15/mt5/live_bot_mt5.py main loop for profile m15 --
         signal from the last CLOSED M15 bar (close beyond the 10-bar channel, H4 EMA30
         trend >= 0.5% away), market entry at the live quote 3 s after the bar opens (or
         after the previous exit: it may re-enter on the same bar), SL = row close -/+ 3*ATR14,
         TP = 3R from the row close, lots rounded to 0.01, 7-day time stop, 2 h pause after
         3 losses in a row. Stops fill at the triggering tick, targets at the TP price.
Both   : fixed sizing equity 25,561.75 USD (no compounding), commission 7 USD/lot round trip,
         swap and the AI review are NOT modelled. Accounts are hedging, so the bots are
         independent. Output: combined_tick_backtest_20260924/.
"""
from pathlib import Path
import sys, json, math
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/"SLP2")); sys.path.insert(0,str(HERE/"M15"))
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from bot.sp2l import DEFAULTS
from scripts.pattern_strategy import load_m15
from scripts.sp2l_tick_backtest import run_ticks, TickStream, TICKS
from strategy.donchian import add_donchian_indicators, add_trend_indicator, trend_direction, donchian_signal

OUT=HERE/"combined_tick_backtest_20260924"
EQUITY=25561.75
RISK_PCT=0.002
COMMISSION=7.0
CONTRACT=100.
BAR=pd.Timedelta(minutes=15)
D_N,D_ATR,D_EMA,D_STRENGTH,D_ATR_MULT,D_RR=10,14,30,0.5,3.0,3.0
D_TIME_STOP=pd.Timedelta(days=7)
D_LATENCY=pd.Timedelta(seconds=3)
D_COOLDOWN_LOSSES,D_COOLDOWN=3,pd.Timedelta(hours=2)


def slp2_dollars(trades):
    budget=EQUITY*RISK_PCT
    rows=[]
    for _,x in trades.iterrows():
        unit_loss=(x.stop_dist+0.20)*CONTRACT+COMMISSION   # worst entry (20-point deviation) to stop, per lot
        lots=math.floor(budget/unit_loss/0.01+1e-9)*0.01
        if lots<0.01:
            continue
        d=1 if x.direction=="long" else -1
        rows.append(dict(bot="SLP2",entry_time=x.entry_time,exit_time=x.exit_time,direction=x.direction,lots=round(lots,2),
            entry=x.entry_price,exit=x.exit_price,reason=x.exit_reason,
            usd=lots*(CONTRACT*d*(x.exit_price-x.entry_price)-COMMISSION)))
    return pd.DataFrame(rows)


def donchian(frame,start,end,trend_lookahead=False):
    """trend_lookahead=True reproduces strategy/donchian.py::simulate_donchian's H4 alignment
    (the H4 bar CONTAINING the signal bar, i.e. its future close) -- diagnostic only."""
    low=add_donchian_indicators(frame.rename(columns={"bar_time":"ts"}),D_N,D_ATR)
    h4=frame.set_index("bar_time")[["open","high","low","close"]].resample("4h",label="left",closed="left").agg(
        {"open":"first","high":"max","low":"min","close":"last"}).dropna()
    h4=add_trend_indicator(h4,D_EMA)
    h4_close=h4.index+pd.Timedelta(hours=4)
    ticks=TickStream(TICKS)
    rows=[]; pos=None; last_exit=None; losses=0; paused_until=None
    first=int(low.ts.searchsorted(start))
    for j in range(max(first,1),len(low)):
        t0=low.ts.iloc[j]
        if t0>=end:
            break
        t1=t0+BAR
        row=low.iloc[j-1]                                    # last CLOSED bar while bar j forms
        if trend_lookahead:
            k=int(np.searchsorted(h4.index.values,row.ts.to_datetime64(),side="right"))-1
        else:
            k=int(np.searchsorted(h4_close.values,t0.to_datetime64(),side="right"))-1
        trend=trend_direction(h4.iloc[k],D_STRENGTH) if k>=0 else "flat"
        signal=donchian_signal(row,trend) if trend!="flat" else None
        cursor=t0
        while True:
            if pos is not None:
                t,b,a=ticks.window(max(cursor,pos["entry_time"]+pd.Timedelta(microseconds=1)),t1)
                if len(t)==0:
                    break
                d=pos["d"]; q=b if d==1 else a
                hits=[(int(np.argmax(m)),r) for m,r in (((q<=pos["sl"]) if d==1 else (q>=pos["sl"]),"stop"),
                        ((q>=pos["tp"]) if d==1 else (q<=pos["tp"]),"target"),(t>=pos["deadline"].value,"time_stop")) if m.any()]
                if not hits:
                    break
                i,reason=min(hits)
                px=pos["tp"] if reason=="target" else float(q[i])
                usd=pos["lots"]*(CONTRACT*d*(px-pos["entry"])-COMMISSION)
                when=pd.Timestamp(t[i])
                rows.append(dict(bot="Donchian",entry_time=pos["entry_time"],exit_time=when,direction="long" if d==1 else "short",
                    lots=pos["lots"],entry=pos["entry"],exit=px,reason=reason,usd=usd))
                losses=0 if usd>0 else losses+1
                if losses>=D_COOLDOWN_LOSSES:
                    paused_until=when+D_COOLDOWN; losses=0
                pos=None; last_exit=when; cursor=when
                continue
            if signal is None:
                break
            ready=max(t0,cursor)+D_LATENCY
            if last_exit is not None:
                ready=max(ready,last_exit+D_LATENCY)
            if paused_until is not None and ready<paused_until:
                ready=paused_until
            if ready>=t1:
                break
            t,b,a=ticks.window(ready,t1)
            if len(t)==0:
                break
            d=1 if signal=="long" else -1
            stop_dist=row.atr*D_ATR_MULT
            sl=row.close-d*stop_dist; tp=row.close+d*stop_dist*D_RR
            entry=float(a[0] if d==1 else b[0]); exit_side=float(b[0] if d==1 else a[0])
            if d*(exit_side-sl)<=0 or d*(tp-exit_side)<=0:
                break                                        # broker would refuse SL/TP on the wrong side
            lots=max(0.01,round(round(EQUITY*RISK_PCT/stop_dist/CONTRACT/0.01)*0.01,2))
            when=pd.Timestamp(t[0])
            pos=dict(d=d,entry=entry,sl=sl,tp=tp,lots=lots,entry_time=when,deadline=when+D_TIME_STOP)
            cursor=when
    return pd.DataFrame(rows)


def stats(df):
    if df.empty:
        return {}
    df=df.sort_values("exit_time"); eq=df.usd.cumsum()
    dd=float((eq.cummax().clip(lower=0)-eq).max())
    return dict(trades=len(df),wins=int((df.usd>0).sum()),win_rate=round(float((df.usd>0).mean()),3),
        net_usd=round(float(df.usd.sum()),2),net_pct=round(float(df.usd.sum())/EQUITY*100,2),
        profit_factor=round(float(df.usd.clip(lower=0).sum()/-df.usd.clip(upper=0).sum()),2),
        max_dd_usd=round(dd,2),max_dd_pct=round(dd/EQUITY*100,2),
        worst_trade_usd=round(float(df.usd.min()),2),best_trade_usd=round(float(df.usd.max()),2))


def main():
    OUT.mkdir(exist_ok=True)
    meta=pq.ParquetFile(TICKS)
    start=pd.Timestamp(meta.read_row_group(0,columns=["time_msc"]).column(0)[0].as_py()).ceil("15min")
    end=pd.Timestamp(meta.read_row_group(meta.metadata.num_row_groups-1,columns=["time_msc"]).column(0)[-1].as_py()).floor("15min")-BAR
    frame=load_m15(); frame=frame[frame.bar_time<end+BAR].reset_index(drop=True)
    slp2=slp2_dollars(run_ticks(frame,start,end)); print("SLP2 done",len(slp2),flush=True)
    don=donchian(frame,start,end); print("Donchian done",len(don),flush=True)
    both=pd.concat([slp2,don],ignore_index=True).sort_values("exit_time")
    both.to_csv(OUT/"all_trades.csv",index=False)
    monthly=both.assign(month=both.exit_time.dt.strftime("%Y-%m")).pivot_table(index="month",columns="bot",values="usd",aggfunc="sum",fill_value=0)
    monthly["total"]=monthly.sum(axis=1)
    overlap=0
    for _,x in slp2.iterrows():
        overlap+=int(((don.entry_time<x.exit_time)&(don.exit_time>x.entry_time)).any())
    report=dict(window=[str(start),str(end)],equity=EQUITY,SLP2=stats(slp2),Donchian=stats(don),combined=stats(both),
                slp2_trades_overlapping_a_donchian_trade=overlap,monthly_usd=monthly.round(2).to_dict("index"))
    (OUT/"report.json").write_text(json.dumps(report,indent=1,default=str),encoding="utf-8")
    print(json.dumps(report,indent=1,default=str))


if __name__=="__main__":
    main()
