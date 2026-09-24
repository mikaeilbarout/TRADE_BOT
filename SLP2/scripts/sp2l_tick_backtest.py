"""SLP2 M15 tick-execution backtest over the 6-month tick file.

Signals: the shared PatternEngine on M15 bars (same rules as live, incl. the
single-position rule). Execution: real bid/ask ticks -- entry at the first tick
LATENCY after the signal bar closes (buy at ask, sell at bid); stop and target
triggered by the exit-side quote (bid for longs, ask for shorts) and filled at that
tick (stops) or at the target price (limits). Commission as in the OHLC model; no
extra slippage (the tick path carries it). The tick file is streamed by row group.
Output: data/sp2l_tick_backtest_20260923/.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from bot import config
from bot.sp2l import DEFAULTS, PatternEngine
from bot.state import atomic_json
from scripts.pattern_strategy import load_m15
from scripts.sp2l_m15_backtest import simulate, summarize

ROOT=Path(__file__).resolve().parents[1]
TICKS=ROOT/"data"/"XAUUSD_ticks_6m.parquet"
OUT=ROOT/"data"/"sp2l_tick_backtest_20260923"
LATENCY=pd.Timedelta(seconds=15)   # live poll interval
BAR=pd.Timedelta(minutes=15)
FEE=config.COMMISSION_PER_LOT/100.  # price units per 1-lot round trip (contract 100)


class TickStream:
    """Forward-only window reader over a large tick parquet (times in int64 ns)."""
    def __init__(self,path):
        self.file=pq.ParquetFile(path)
        self.next_group=0
        self.t=np.empty(0,dtype=np.int64); self.bid=np.empty(0); self.ask=np.empty(0)

    def _load_until(self,t_end):
        while (len(self.t)==0 or self.t[-1]<t_end) and self.next_group<self.file.metadata.num_row_groups:
            g=self.file.read_row_group(self.next_group,columns=["time_msc","bid","ask"])
            self.next_group+=1
            self.t=np.concatenate([self.t,g.column("time_msc").to_numpy().astype("datetime64[ns]").astype(np.int64)])
            self.bid=np.concatenate([self.bid,g.column("bid").to_numpy()])
            self.ask=np.concatenate([self.ask,g.column("ask").to_numpy()])

    def window(self,start,end):
        s,e=pd.Timestamp(start).value,pd.Timestamp(end).value
        self._load_until(e)
        lo=int(np.searchsorted(self.t,s,"left"))
        if lo>0:  # forget everything before the window (reads are chronological)
            self.t,self.bid,self.ask=self.t[lo:],self.bid[lo:],self.ask[lo:]
            lo=0
        hi=int(np.searchsorted(self.t,e,"left"))
        return self.t[lo:hi],self.bid[lo:hi],self.ask[lo:hi]


def trend_age_blocks(engine,skip):
    if skip is None:
        return None
    c=engine.c
    side=np.sign(c-pd.Series(c).ewm(span=200,adjust=False).mean().to_numpy())
    run=np.zeros(len(c),dtype=int)
    for k in range(1,len(c)):
        run[k]=run[k-1]+1 if side[k]==side[k-1] else 0
    return side,run


def run_ticks(frame,start,end,skip=None,params=DEFAULTS,entry_filter=None,early_exit=None,
              bar_minutes=15,engine_factory=None):
    """Bars cover warm-up history; only signals decided in [start,end) are traded."""
    BAR=pd.Timedelta(minutes=bar_minutes)
    engine=(engine_factory or PatternEngine)(frame,params)
    times=pd.to_datetime(engine.frame.bar_time)
    trend=trend_age_blocks(engine,skip)
    keep=entry_filter(engine.frame) if entry_filter is not None else None
    ticks=TickStream(TICKS)
    rows,pending,queued,trade=[],None,None,None
    first=int(times.searchsorted(start-pd.Timedelta(days=2)))  # pattern state needs a little lead-in
    for i in range(first,len(times)):
        bar_open=times.iloc[i]
        if bar_open>=end:
            break
        busy=trade is not None
        if queued is not None and bar_open==queued["decision"]:
            t,b,a=ticks.window(bar_open+LATENCY,bar_open+BAR)
            if len(t):
                d=queued["direction"]; entry=a[0] if d==1 else b[0]; exit_quote=b[0] if d==1 else a[0]
                distance=d*(entry-queued["stop"])
                if 0<distance<=queued.get("max_stop",params.max_stop) and d*(exit_quote-queued["stop"])>0:
                    trade=dict(direction=d,entry=entry,stop=queued["stop"],target=entry+d*params.rr*distance,
                               distance=distance,entry_time=pd.Timestamp(t[0]),signal=queued["decision"],
                               deadline=pd.Timestamp(t[0])+pd.Timedelta(minutes=params.max_hold_minutes))
                    busy=True
            queued=None
        if trade is not None and trade.pop("exit_now",False):
            t,b,a=ticks.window(bar_open+LATENCY,bar_open+BAR)
            if len(t):
                d=trade["direction"]; px=b[0] if d==1 else a[0]; pnl=d*(px-trade["entry"])-FEE
                rows.append(dict(signal_time=trade["signal"],entry_time=trade["entry_time"],exit_time=pd.Timestamp(t[0]),
                    direction="long" if d==1 else "short",entry_price=trade["entry"],exit_price=px,stop_price=trade["stop"],
                    target_price=trade["target"],stop_dist=trade["distance"],pnl=pnl,r_multiple=pnl/(trade["distance"]+FEE),
                    exit_reason="early_exit"))
                trade=None
                busy=True
            else:
                trade["exit_now"]=True  # no quotes yet; try on the next bar
        if trade is not None:
            t,b,a=ticks.window(max(bar_open,trade["entry_time"]+pd.Timedelta(microseconds=1)),bar_open+BAR)
            if len(t):
                d=trade["direction"]; q=b if d==1 else a
                stop_hit=(q<=trade["stop"]) if d==1 else (q>=trade["stop"])
                tgt_hit=(q>=trade["target"]) if d==1 else (q<=trade["target"])
                late=t>=trade["deadline"].value
                hits=[(np.argmax(m),r) for m,r in ((stop_hit,"stop"),(tgt_hit,"target"),(late,"time")) if m.any()]
                if hits:
                    k,reason=min(hits,key=lambda h:h[0])
                    px=trade["target"] if reason=="target" else q[k]
                    pnl=d*(px-trade["entry"])-FEE
                    rows.append(dict(signal_time=trade["signal"],entry_time=trade["entry_time"],exit_time=pd.Timestamp(t[k]),
                        direction="long" if d==1 else "short",entry_price=trade["entry"],exit_price=px,stop_price=trade["stop"],
                        target_price=trade["target"],stop_dist=trade["distance"],pnl=pnl,r_multiple=pnl/(trade["distance"]+FEE),
                        exit_reason=reason))
                    trade=None
            if trade is not None and early_exit is not None and len(t):
                trade["mfe"]=max(trade.get("mfe",0.),float(d*(q.max() if d==1 else q.min())-d*trade["entry"])/trade["distance"])
                # Count from the entry BAR (signal decision time), as the OHLC model does.
                if bar_open+BAR-trade["signal"]>=pd.Timedelta(hours=early_exit[0]) and trade["mfe"]<early_exit[1]:
                    trade["exit_now"]=True
        if busy:
            pending=None
            continue
        pending,trigger=engine.advance(pending,i)
        if trigger is None:
            continue
        decision=pd.Timestamp(trigger["bar_time"])+BAR
        if not (start<=decision<end):
            continue
        if trend is not None:
            side,run=trend
            if side[i]==trigger["direction"] and skip[0]<=run[i]*15/1440<skip[1]:
                continue
        if keep is not None and not keep(i,trigger["direction"]):
            continue
        queued=dict(direction=trigger["direction"],stop=trigger["stop"],decision=decision)
        if "max_stop" in trigger:
            queued["max_stop"]=trigger["max_stop"]
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    meta=pq.ParquetFile(TICKS)
    first=meta.read_row_group(0,columns=["time_msc"]).column(0)[0].as_py()
    last=meta.read_row_group(meta.metadata.num_row_groups-1,columns=["time_msc"]).column(0)[-1].as_py()
    start=pd.Timestamp(first).ceil("15min"); end=pd.Timestamp(last).floor("15min")-BAR
    frame=load_m15()
    frame=frame[frame.bar_time<end+BAR].reset_index(drop=True)
    report=dict(tick_window=[str(start),str(end)],latency_seconds=LATENCY.total_seconds(),rr=DEFAULTS.rr)
    for name,skip in (("no_filter",None),("skip_1_10d",(1.,10.))):
        ticks=run_ticks(frame,start,end,skip)
        ticks.to_csv(OUT/f"{name}_tick_trades.csv",index=False)
        ohlc=simulate(frame,skip_trend_age_days=skip,evaluation_start=start)
        ohlc=ohlc[ohlc.signal_time<end]
        ohlc.to_csv(OUT/f"{name}_ohlc_trades.csv",index=False)
        report[name]=dict(tick=summarize(ticks),ohlc=summarize(ohlc),
                          tick_exits=ticks.exit_reason.value_counts().to_dict() if len(ticks) else {})
        print(name,json.dumps(report[name],default=str),flush=True)
    atomic_json(OUT/"report.json",report)


if __name__=="__main__":
    main()
