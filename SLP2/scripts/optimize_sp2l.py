"""Deterministic bounded search; final 30% never participates in ranking."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json, hashlib, random
from dataclasses import asdict
import numpy as np
from bot.sp2l import DEFAULTS
from bot.state import atomic_json
from scripts.pattern_strategy import load_m15
from scripts.sp2l_m15_backtest import simulate, summarize

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data'/'sp2l_optimization_20260923'

def kwargs(p):
    return dict(rr=p['rr'],p_gap_price=p['gap'],max_sl_dist=p['max_stop'],spike_size=p['spike_size'],
        use_ema_filter=p['ema_filter'],ema_period=p['ema_period'],use_trend_filter=p['trend_filter'],
        max_opposite_moves=p['max_opposite'],max_hold_bars=p['max_hold_minutes']//15)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    frame=load_m15()
    split=int(len(frame)*.7)
    train=frame.iloc[:split].reset_index(drop=True)
    boundary=frame.bar_time.iloc[split]
    base=asdict(DEFAULTS)
    rng=random.Random(23092026)
    candidates=[base,base|dict(rr=1.,gap=1.,ema_period=60,max_opposite=1)]
    space=dict(rr=[1.,1.5,2.,2.5,3.,4.],gap=[.5,1.,1.5,2.],max_stop=[6.,10.,15.],
        spike_size=[1.25,1.5,2.],ema_period=[20,40,60,100],ema_filter=[True,False],max_opposite=[1,2,3])
    while len(candidates)<98:
        p=base|{k:rng.choice(v) for k,v in space.items()}
        if not p['ema_filter']: p['ema_period']=20
        if p not in candidates:candidates.append(p)
    protocol=dict(seed=23092026,candidates=candidates,split_time=str(boundary),
        ranking='train only: total_R / max(5, drawdown_R), minus 0.5 per negative temporal third; minimum 10 trades per third',
        promotion='winner only, no fallback search: holdout PF>1, total_R>baseline, DD<=1.5*baseline, trades>=40',
        caveat='Historical final 30% already inspected in earlier tasks; not fresh untouched out-of-sample evidence',
        costs='historical spread, 2-point slippage, commission 7 per lot, no swap',
        source_hash=hashlib.sha256((ROOT/'scripts/sp2l_m15_backtest.py').read_bytes()).hexdigest())
    atomic_json(OUT/'protocol.json',protocol)
    rows=[]
    boundaries=[train.bar_time.iloc[int(len(train)*j/3)] for j in (1,2)]
    for idx,p in enumerate(candidates):
        trades=simulate(train,**kwargs(p))
        stats=summarize(trades)
        masks=[trades.entry_time<boundaries[0],(trades.entry_time>=boundaries[0])&(trades.entry_time<boundaries[1]),trades.entry_time>=boundaries[1]]
        blocks=[summarize(trades[m]) for m in masks]
        eligible=all(b['n_trades']>=10 for b in blocks)
        score=stats['total_r']/max(5.,stats['max_dd_r'])-.5*sum(b['total_r']<0 for b in blocks) if eligible else None
        rows.append(dict(id=idx,parameters=p,train=stats,blocks=blocks,score=score))
        if idx%10==0: print(f"Completed {idx+1}/{len(candidates)}",flush=True)
    ranked=sorted((r for r in rows if r['score'] is not None),key=lambda r:r['score'],reverse=True)
    if not ranked:raise RuntimeError('No candidate meets minimum trade counts')
    winner=ranked[0]
    results={}
    for label,p in [('baseline',base),('selected',winner['parameters'])]:
        warmup=max(250,p['ema_period']*10+4)
        test=frame.iloc[split-warmup:].reset_index(drop=True)
        trades=simulate(test,evaluation_start=boundary,**kwargs(p))
        trades.to_csv(OUT/f'{label}_test_trades.csv',index=False)
        stress=simulate(test,evaluation_start=boundary,slippage_points=5,commission_per_lot=10.,**kwargs(p))
        results[label]=dict(parameters=p,test=summarize(trades),stress_test=summarize(stress))
    a,b=results['selected']['test'],results['baseline']['test']
    promote=a['n_trades']>=40 and (a['profit_factor'] or 0)>1 and a['total_r']>b['total_r'] and a['max_dd_r']<=1.5*b['max_dd_r']
    report=dict(protocol=protocol,ranked_train=ranked,all_candidates=rows,results=results,promotion_passed=promote)
    atomic_json(OUT/'report.json',report)
    print(json.dumps(dict(winner=winner,results=results,promotion_passed=promote),indent=2),flush=True)

if __name__=='__main__':main()
