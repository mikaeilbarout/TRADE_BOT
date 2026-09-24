"""Causal M15 SP2L backtest sharing the live pattern engine.

Signals are known at candle close. Fills use the NEXT candle open with Bid/Ask
spread, never the completed trigger candle low/high. OHLC execution remains
an approximation, with stop-first tie breaks and explicit end-of-data exits.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
from dataclasses import asdict
import json
import numpy as np
import pandas as pd
from bot import config
from bot.sp2l import DEFAULTS,Parameters,PatternEngine,add_indicators,detect_buy_setup,detect_sell_setup
from bot.state import atomic_json
from scripts.pattern_strategy import load_m15

SPIKE_CANDLE_SIZE=DEFAULTS.spike_size
POINT=.01
PGAP_POINTS=DEFAULTS.gap/POINT
MAX_SL_DISTANCE_POINTS=DEFAULTS.max_stop/POINT
TP_R=DEFAULTS.rr
USE_EMA_FILTER,EMA_PERIOD=DEFAULTS.ema_filter,DEFAULTS.ema_period
USE_TREND_FILTER,MAX_OPPOSITE_MOVES=DEFAULTS.trend_filter,DEFAULTS.max_opposite
COLUMNS=["signal_time","entry_time","exit_time","direction","entry_price","exit_price",
         "stop_price","target_price","stop_dist","pnl","pnl_cash_per_lot","r_multiple","exit_reason"]


def simulate(df,rr=TP_R,p_gap_price=PGAP_POINTS*POINT,max_sl_dist=MAX_SL_DISTANCE_POINTS*POINT,
             spike_size=SPIKE_CANDLE_SIZE,use_ema_filter=USE_EMA_FILTER,ema_period=EMA_PERIOD,
             use_trend_filter=USE_TREND_FILTER,max_opposite_moves=MAX_OPPOSITE_MOVES,
             max_hold_bars=500,apply_costs=True,*,slippage_points=2,point_size=POINT,
             contract_size=100.,commission_per_lot=config.COMMISSION_PER_LOT,
             evaluation_start=None,bar_minutes=15,session_hours=None,lock_steps=None,max_positions=1,
             skip_trend_age_days=None,trend_ema=200,entry_filter=None,early_exit=None,weekend_close_hour=None,
             engine_factory=None):
    """lock_steps: ((reach_R, lock_R), ...). After a bar CLOSES with the trade having
    reached reach_R, the stop moves to lock_R (0 = entry). Never applied intrabar.
    max_positions: concurrent trades allowed; pattern scanning pauses (and pending
    resets) on any bar where that many trades were open, as with a single position."""
    if not isinstance(max_positions,int) or max_positions<1:
        raise ValueError("max_positions must be a positive integer")
    # skip_trend_age_days=(lo,hi): drop a trigger when the close has been on the trade's
    # side of EMA(trend_ema) for lo <= days < hi (a move already under way).
    # entry_filter: factory(frame) -> keep(i, direction); called on each trigger bar i.
    # early_exit=(hours, min_R): at a bar close at least `hours` after entry, if the best
    # closed-bar excursion is still below min_R, exit at the NEXT bar's open.
    # weekend_close_hour=h: on Friday (data clock = broker server time) from hour h, open
    # trades exit at the bar's open and no new entries are taken until the week reopens.
    if not isinstance(max_hold_bars,int) or max_hold_bars<1:
        raise ValueError("max_hold_bars must be a positive integer")
    if not all(np.isfinite(v) and v>=0 for v in (slippage_points,commission_per_lot)) or not all(np.isfinite(v) and v>0 for v in (point_size,contract_size)):
        raise ValueError("Invalid cost model")
    params=Parameters(spike_size,p_gap_price,max_sl_dist,rr,use_ema_filter,ema_period,
                      use_trend_filter,max_opposite_moves,max_hold_bars*bar_minutes)
    bar=pd.Timedelta(minutes=bar_minutes)
    # engine_factory(df, params) -> engine with the PatternEngine interface; a trigger may
    # carry its own "max_stop" (e.g. volatility-scaled), otherwise max_sl_dist applies.
    engine=(engine_factory or PatternEngine)(df,params)
    frame=engine.frame
    n=len(frame)
    if apply_costs:
        if "avg_spread_price" in frame:
            spread=frame.avg_spread_price.to_numpy(dtype=float)
        elif "spread" in frame:
            spread=frame.spread.to_numpy(dtype=float)*point_size
        else:
            raise ValueError("Costed simulation requires avg_spread_price or broker spread points")
        if not np.isfinite(spread).all() or (spread<0).any():
            raise ValueError("Invalid historical spread")
    else:
        spread=np.zeros(n)
    fee=commission_per_lot/contract_size if apply_costs else 0.
    slip=slippage_points*point_size if apply_costs else 0.
    rows,pending,queued,open_trades=[],None,None,[]
    times=pd.to_datetime(frame.bar_time).to_numpy()
    keep=entry_filter(frame) if entry_filter is not None else None
    if skip_trend_age_days is not None:
        trend_side=np.sign(engine.c-pd.Series(engine.c).ewm(span=trend_ema,adjust=False).mean().to_numpy())
        side_run=np.zeros(n,dtype=int)  # consecutive closed bars on the current side, ending at i
        for k in range(1,n):
            side_run[k]=side_run[k-1]+1 if trend_side[k]==trend_side[k-1] else 0

    def close_trade(trade,px,when,reason):
        pnl=trade["direction"]*(px-trade["entry"])-fee
        rows.append(dict(signal_time=trade["signal_time"],entry_time=trade["entry_time"],exit_time=when,
            direction="long" if trade["direction"]==1 else "short",entry_price=trade["entry"],exit_price=px,
            stop_price=trade["stop"],target_price=trade["target"],stop_dist=trade["distance"],
            pnl=pnl,pnl_cash_per_lot=pnl*contract_size,r_multiple=pnl/(trade["distance"]+fee),exit_reason=reason))

    for i in range(n):
        stamp=pd.Timestamp(times[i])
        busy=len(open_trades)
        if queued is not None:
            d,sl=queued["direction"],queued["stop"]
            delay=(stamp-pd.Timestamp(queued["decision_time"])).total_seconds()/60
            entry=engine.o[i]+(spread[i] if d==1 else 0)+d*slip
            distance=d*(entry-sl)
            exit_quote=engine.o[i]+(spread[i] if d==-1 else 0.)
            stop_is_executable=d*(exit_quote-sl)>=point_size-1e-10
            if 0<=delay<=params.max_entry_delay_minutes and 0<distance<=queued.get("max_stop",max_sl_dist) and stop_is_executable:
                open_trades.append(dict(direction=d,entry=entry,stop=sl,initial_stop=sl,target=entry+d*rr*distance,distance=distance,
                    entry_time=stamp,signal_time=pd.Timestamp(queued["decision_time"]),
                    deadline=stamp+pd.Timedelta(minutes=params.max_hold_minutes)))
                busy+=1
            queued=None
        for trade in list(open_trades):
            d=trade["direction"]
            extra=spread[i] if d==-1 else 0.
            opening,high,low=engine.o[i]+extra,engine.h[i]+extra,engine.l[i]+extra
            stop,target=trade["stop"],trade["target"]
            if trade.pop("exit_now",False):
                close_trade(trade,opening-d*slip,stamp,"early_exit")
                open_trades.remove(trade)
                continue
            if weekend_close_hour is not None and stamp.weekday()==4 and stamp.hour>=weekend_close_hour:
                close_trade(trade,opening-d*slip,stamp,"weekend")
                open_trades.remove(trade)
                continue
            if d*(opening-stop)<=0:
                close_trade(trade,opening-d*slip,stamp,"stop_gap")
            elif d*(opening-target)>=0:
                close_trade(trade,target,stamp,"target_gap")
            elif stamp>=trade["deadline"]:
                close_trade(trade,opening-d*slip,stamp,"time")
            elif (low<=stop if d==1 else high>=stop):
                close_trade(trade,stop-d*slip,stamp+bar,"locked_stop" if stop!=trade["initial_stop"] else "stop")
            elif (high>=target if d==1 else low<=target):
                close_trade(trade,target,stamp+bar,"target")
            else:
                trade["open"]=True
            if not trade.pop("open",False):
                open_trades.remove(trade)
                continue
            if early_exit is not None:
                trade["mfe"]=max(trade.get("mfe",0.),d*((high if d==1 else low)-trade["entry"])/trade["distance"])
                if stamp+bar-trade["entry_time"]>=pd.Timedelta(hours=early_exit[0]) and trade["mfe"]<early_exit[1]:
                    trade["exit_now"]=True
            if lock_steps:
                reached=d*((high if d==1 else low)-trade["entry"])/trade["distance"]
                for reach,lock in lock_steps:
                    candidate=trade["entry"]+d*lock*trade["distance"]
                    if reached>=reach and d*(candidate-trade["stop"])>0:
                        trade["stop"]=candidate
        if busy>=max_positions:
            pending=None
            continue
        pending,trigger=engine.advance(pending,i)
        if trigger is not None:
            # The shared engine stamps M15 decisions; a signal is known when THIS bar closes.
            trigger["decision_time"]=str(pd.Timestamp(trigger["bar_time"])+bar)
            # Optional [start,end) hour window on the data's own clock (broker server time).
            if session_hours is not None:
                hour=pd.Timestamp(trigger["decision_time"]).hour
                start,end=session_hours
                if not (start<=hour<end if start<end else hour>=start or hour<end):
                    trigger=None
            if trigger is not None and skip_trend_age_days is not None and trend_side[i]==trigger["direction"]:
                age_days=side_run[i]*bar_minutes/1440
                if skip_trend_age_days[0]<=age_days<skip_trend_age_days[1]:
                    trigger=None
            if trigger is not None and keep is not None and not keep(i,trigger["direction"]):
                trigger=None
            if trigger is not None and weekend_close_hour is not None:
                entry_at=pd.Timestamp(trigger["decision_time"])
                if entry_at.weekday()==4 and entry_at.hour>=weekend_close_hour:
                    trigger=None
        if trigger is not None:
            if evaluation_start is None or pd.Timestamp(trigger["decision_time"])>=pd.Timestamp(evaluation_start):
                queued=trigger
    for trade in open_trades:
        d=trade["direction"]
        px=engine.c[-1]+(spread[-1] if d==-1 else 0)-d*slip
        close_trade(trade,px,pd.Timestamp(times[-1])+bar,"end_of_data")
    result=pd.DataFrame(rows,columns=COLUMNS)
    for col in ("signal_time","entry_time","exit_time"):
        result[col]=pd.to_datetime(result[col])
    return result


def summarize(trades):
    if trades.empty:
        return dict(n_trades=0,win_rate=None,avg_r=None,profit_factor=None,total_r=0.,max_dd_r=0.,recovery_factor=None)
    ordered=trades.sort_values("exit_time")
    r=ordered.r_multiple.astype(float)
    if not np.isfinite(r).all():
        raise ValueError("Non-finite trade return")
    cumulative=r.cumsum()
    dd=float((cumulative.cummax().clip(lower=0)-cumulative).max())
    gain,loss=float(r.clip(lower=0).sum()),float(-r.clip(upper=0).sum())
    return dict(n_trades=len(r),win_rate=float((r>0).mean()),avg_r=float(r.mean()),
        profit_factor=gain/loss if loss>0 else None,total_r=float(r.sum()),max_dd_r=dd,
        recovery_factor=float(r.sum())/dd if dd>0 else None,
        end_of_data_exits=int((ordered.exit_reason=="end_of_data").sum()) if "exit_reason" in ordered else 0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split",type=float,default=.70)
    args=parser.parse_args()
    if not 0<args.split<1:
        parser.error("split must be between zero and one")
    frame=load_m15()
    split=int(len(frame)*args.split)
    if split<2 or split>=len(frame)-1:
        raise ValueError("Insufficient history for split")
    train=frame.iloc[:split].reset_index(drop=True)
    boundary=frame.bar_time.iloc[split]
    warmup=max(250,DEFAULTS.ema_period*10+4)
    test=frame.iloc[max(0,split-warmup):].reset_index(drop=True)
    trades_train=simulate(train)
    trades_test=simulate(test,evaluation_start=boundary)
    root=Path(__file__).resolve().parents[1]
    # data/slp2_reviewed_20260923 is the frozen RR=3 review record; never overwrite it.
    out=root/"data"/"sp2l_m15_latest"
    out.mkdir(parents=True,exist_ok=True)
    trades_train.to_csv(out/"train_trades.csv",index=False)
    trades_test.to_csv(out/"test_trades.csv",index=False)
    report=dict(parameters=asdict(DEFAULTS),split=args.split,split_time=str(boundary),
        data_start=str(frame.bar_time.iloc[0]),data_end=str(frame.bar_time.iloc[-1]),
        model="causal next-open Bid/Ask OHLC approximation; not tick validated",
        validation="retrospective diagnostic; previous parameter-selection provenance unavailable",
        assumptions=dict(commission_per_lot=config.COMMISSION_PER_LOT,slippage_points=2,contract_size=100.,swap_included=False),
        train=summarize(trades_train),test=summarize(trades_test))
    atomic_json(out/"report.json",report)
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
