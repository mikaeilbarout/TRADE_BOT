"""SLP2 M15 runner. Shared causal pattern engine; dry-run unless --live.

Historical validation claims from the previous source do not validate the
corrected market-entry model. See audit/review_20260923/REVIEW_FA.md.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import logging
import math
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from logging.handlers import RotatingFileHandler
from pathlib import Path
import time
import MetaTrader5 as mt5
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

from bot import config
from bot import notifier
from bot.ai_review import log_agent_detail, report_outcome, review_signal
from bot.mt5_data import (get_bars, get_tick, account, identity, positions, get_symbol_info, utc_now,
    refresh_server_offset, server_ms_to_utc, server_seconds_to_utc_epoch, utc_to_server_datetime)
from bot.risk import lots_for_risk, validate_risk
from bot.sp2l import DEFAULTS, Parameters, PatternEngine, replay
from bot.state import atomic_json, acquire_lock as lock_file

SYMBOL = "XAUUSD"
SPIKE_CANDLE_SIZE, PGAP_DOLLARS = DEFAULTS.spike_size, DEFAULTS.gap
MAX_SL_DOLLARS, TP_R = DEFAULTS.max_stop, DEFAULTS.rr
USE_EMA_FILTER, EMA_PERIOD = DEFAULTS.ema_filter, DEFAULTS.ema_period
USE_TREND_FILTER, MAX_OPPOSITE_MOVES = DEFAULTS.trend_filter, DEFAULTS.max_opposite
RISK_PCT = .002
COOLDOWN_LOSSES_TO_TRIGGER = 3
COOLDOWN_HOURS = 2.0
MAX_DAILY_LOSS_PCT = 100.0  # same live-M15 setting: disabled unless deliberately tightened
MAGIC = 20260223
DEVIATION_POINTS = 20
MAX_SIGNAL_AGE_MIN = DEFAULTS.max_entry_delay_minutes  # measured from candle CLOSE
# Optional: once a CLOSED M15 bar has reached this many R, move our stop to entry
# (backtest: lock_steps=((R,0),)). Disabled by choice 2026-09-23: 2.0 cut drawdown
# 12.2R -> 10.3R but profit 58.7R -> 48.3R. Evidence: data/sp2l_profit_lock_20260923/,
# data/sp2l_4segments_20260923/.
BREAKEVEN_AT_R = None
BARS_NEEDED = max(250, EMA_PERIOD*10+4)
POLL_SECONDS = 15
RUNTIME_DIR = ROOT/"runtime"
LOG_PATH = RUNTIME_DIR/"slp2.log"
LOCK_PATH = RUNTIME_DIR/"slp2.lock"
# Only positive evidence permits retry after a send attempt.
REJECTED_CODES = {10004,10006,10013,10014,10015,10016,10017,10018,10019,10020,10021,10022,10024,10030}


def strategy_parameters():
    return Parameters(spike_size=SPIKE_CANDLE_SIZE,gap=PGAP_DOLLARS,max_stop=MAX_SL_DOLLARS,
        rr=TP_R,ema_filter=USE_EMA_FILTER,ema_period=EMA_PERIOD,trend_filter=USE_TREND_FILTER,
        max_opposite=MAX_OPPOSITE_MOVES,max_entry_delay_minutes=MAX_SIGNAL_AGE_MIN)


def fingerprint():
    payload=asdict(strategy_parameters()) | {"risk":RISK_PCT,"symbol":SYMBOL,"magic":MAGIC,"breakeven_at_r":BREAKEVEN_AT_R,
        "commission":config.COMMISSION_PER_LOT,"deviation":DEVIATION_POINTS}
    payload["source"]={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in (Path(__file__),ROOT/"bot"/"sp2l.py",ROOT/"bot"/"risk.py",ROOT/"bot"/"bars.py",
                                 ROOT/"bot"/"mt5_data.py")}
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()


def state_path(account_id,live):
    key=hashlib.sha256(account_id.encode()).hexdigest()[:20]
    return RUNTIME_DIR/f"slp2_v2_{key}_{'live' if live else 'dry'}.json"


def load_state(account_id,live):
    path=state_path(account_id,live)
    if not path.exists():
        return {"version":3,"account":account_id,"live":live,"config":fingerprint(),
                "last_bar":None,"pending":None,"candidate":None,"actions":{},"closes":{},"deals":{},"locks":{},
                "outcomes":{},"guards":{"day":None,"day_start_equity":None,"consecutive_losses":0,
                "cooldown_until":None}}
    # Corrupt or context-mismatched state must never be treated as a fresh start.
    state=json.loads(path.read_text(encoding="utf-8"))
    if state.get("version")==2:
        state["version"]=3
        state.setdefault("outcomes",{})
        state.setdefault("guards",{"day":None,"day_start_equity":None,"consecutive_losses":0,"cooldown_until":None})
    required={"version","account","live","config","last_bar","pending","candidate","actions","closes","deals","outcomes","guards"}
    if not required.issubset(state) or state["version"]!=3 or state["account"]!=account_id or state["live"]!=live:
        raise RuntimeError("Invalid/mismatched SLP2 checkpoint; inspect it before trading")
    state.setdefault("locks",{})
    if state["config"]!=fingerprint():
        # Retain the order journal; only pattern state is invalidated by a config change.
        state.update(config=fingerprint(),last_bar=None,pending=None,candidate=None)
    return state


def save_state(state):
    atomic_json(state_path(state["account"],state["live"]),state)


def acquire_lock():
    return lock_file(LOCK_PATH)


def latest_pending_and_trigger(df):
    """Flat research scan only. Live uses persistent incremental pattern state."""
    return replay(df,strategy_parameters())


def filling_mode(info):
    if info.filling_mode & 1:
        return mt5.ORDER_FILLING_FOK
    if info.filling_mode & 2:
        return mt5.ORDER_FILLING_IOC
    if info.trade_exemode!=mt5.SYMBOL_TRADE_EXECUTION_MARKET:
        return mt5.ORDER_FILLING_RETURN
    raise RuntimeError("No supported filling policy")


def quantize_price(price,info,up):
    if not math.isfinite(info.trade_tick_size) or info.trade_tick_size<=0 or not math.isfinite(price) or price<=0:
        raise ValueError("Invalid price/tick size")
    step=Decimal(str(info.trade_tick_size))
    return float((Decimal(str(price))/step).to_integral_value(rounding=ROUND_CEILING if up else ROUND_FLOOR)*step)


def hedging_account():
    return account().margin_mode==getattr(mt5,"ACCOUNT_MARGIN_MODE_RETAIL_HEDGING",2)


def own_exposure(items):
    """On hedging accounts other EAs' tickets are independent; netting shares one position per symbol."""
    return [x for x in items if x.magic==MAGIC] if hedging_account() else list(items)


def has_open_position(logger=None):
    return bool(own_exposure(positions(SYMBOL)))


def check_context(state,live):
    if identity()!=state["account"] or state["live"]!=live:
        raise RuntimeError("Account/mode changed; no order permitted")


def submission_status(result):
    if result is None:
        return "unknown"
    if result.retcode in (mt5.TRADE_RETCODE_DONE,mt5.TRADE_RETCODE_DONE_PARTIAL):
        return "accepted"
    return "rejected" if result.retcode in REJECTED_CODES else "unknown"


def result_fields(result):
    return {key:getattr(result,key,None) for key in ("retcode","order","deal","price","volume")} if result is not None else {}


def unresolved(state):
    return any(v["status"] in ("pending","unknown") for v in state["actions"].values())


def entries_allowed(state, equity, now):
    """Live-M15 daily/cooldown guards, persisted across restarts."""
    guards=state["guards"]
    day=str(now.date())
    if guards["day"]!=day:
        guards.update(day=day,day_start_equity=equity,consecutive_losses=0,cooldown_until=None)
    until=guards.get("cooldown_until")
    if until and now < pd.Timestamp(until):
        return False, f"cooldown until {until}"
    start=guards.get("day_start_equity")
    if start and MAX_DAILY_LOSS_PCT<100 and (start-equity)/start*100>=MAX_DAILY_LOSS_PCT:
        return False, "daily loss limit reached"
    return True, ""


def record_outcome(state, profit, closed_at):
    guards=state["guards"]
    if profit<0:
        guards["consecutive_losses"]+=1
        if guards["consecutive_losses"]>=COOLDOWN_LOSSES_TO_TRIGGER:
            guards["cooldown_until"]=str(closed_at+pd.Timedelta(hours=COOLDOWN_HOURS))
            guards["consecutive_losses"]=0
    else:
        guards["consecutive_losses"]=0


def place_order(trigger,live,logger,state=None):
    state=load_state(identity(),live) if state is None else state
    check_context(state,live)
    token="slp2:"+pd.Timestamp(trigger["bar_time"]).strftime("%Y%m%d%H%M")
    old=state["actions"].get(token,{})
    if old.get("status") in ("accepted","dry_run") or old.get("attempts",0)>=3:
        return "consumed"
    if unresolved(state):
        raise RuntimeError("Unresolved submission; inspect broker history before further orders")
    decision=pd.Timestamp(trigger.get("decision_time",pd.Timestamp(trigger["bar_time"])+pd.Timedelta(minutes=15)))
    age=(utc_now()-decision).total_seconds()/60
    if not 0 <= age <= MAX_SIGNAL_AGE_MIN:
        return "skipped"
    if has_open_position(logger):
        return "skipped"
    orders=mt5.orders_get(symbol=SYMBOL)
    if orders is None:
        raise RuntimeError("Cannot determine outstanding orders")
    if own_exposure(orders):
        return "skipped"
    d=trigger["direction"]
    if d not in (-1,1):
        raise ValueError("Invalid direction")
    info,tick=get_symbol_info(SYMBOL),get_tick(SYMBOL)
    entry=tick.ask if d==1 else tick.bid
    sl=quantize_price(trigger["stop"],info,up=d==-1)
    distance=d*(entry-sl)
    if not 0 < distance <= MAX_SL_DOLLARS:
        return "skipped"
    # Derive reward from the obtainable quote, not the past candle extreme.
    tp=quantize_price(entry+d*TP_R*distance,info,up=d==1)
    reference=tick.bid if d==1 else tick.ask
    minimum=max(info.trade_stops_level*info.point,info.trade_tick_size)
    if d*(reference-sl)<minimum or d*(tp-reference)<minimum:
        return "skipped"
    acc=account()
    allowed,reason=entries_allowed(state,acc.equity,utc_now())
    if not allowed:
        logger.info("Entry blocked by live guard: %s",reason)
        save_state(state)
        return "guard_blocked"
    worst_entry=entry+d*DEVIATION_POINTS*info.point
    lots=lots_for_risk(SYMBOL,abs(worst_entry-sl),RISK_PCT,direction=d,
                       entry=worst_entry,stop=sl,equity=acc.equity)
    if lots<=0:
        return "skipped"
    review=None
    if live:
        review=review_signal(symbol=SYMBOL,side="long" if d==1 else "short",entry=entry,
            stop_loss=sl,take_profit=tp,volume=lots,timeframe="M15",strategy="slp2_m15",
            equity=acc.equity,recent_loss_streak=state["guards"]["consecutive_losses"],
            market_open=True)  # reaching here required a fresh, non-stale live tick just above
        logger.info("AI review: %s for %s -- %s",review.decision,token,review.reason)
        if not review.approved:
            state["actions"][token]={"status":"ai_rejected","decision":review.decision,
                "reason":review.reason,"signal_id":review.signal_id}
            save_state(state)
            return "ai_rejected"
        log_agent_detail(review.signal_id)
        # The review call can block for minutes; entry/sl/tp/lots above were derived
        # from a quote taken BEFORE it. Re-quote and re-validate against the same
        # guards rather than submitting a stale price/size to the broker.
        tick=get_tick(SYMBOL)
        entry=tick.ask if d==1 else tick.bid
        sl=quantize_price(trigger["stop"],info,up=d==-1)
        distance=d*(entry-sl)
        if not 0 < distance <= MAX_SL_DOLLARS:
            return "skipped"
        tp=quantize_price(entry+d*TP_R*distance,info,up=d==1)
        reference=tick.bid if d==1 else tick.ask
        if d*(reference-sl)<minimum or d*(tp-reference)<minimum:
            return "skipped"
        worst_entry=entry+d*DEVIATION_POINTS*info.point
        lots=lots_for_risk(SYMBOL,abs(worst_entry-sl),RISK_PCT,direction=d,
                           entry=worst_entry,stop=sl,equity=acc.equity)
        if lots<=0:
            return "skipped"
    request=dict(action=mt5.TRADE_ACTION_DEAL,symbol=SYMBOL,volume=lots,
        type=mt5.ORDER_TYPE_BUY if d==1 else mt5.ORDER_TYPE_SELL,sl=sl,tp=tp,
        deviation=DEVIATION_POINTS,magic=MAGIC,comment=token,
        type_time=mt5.ORDER_TIME_GTC,type_filling=filling_mode(info))
    if info.trade_exemode!=mt5.SYMBOL_TRADE_EXECUTION_MARKET:
        request["price"]=entry
    margin=mt5.order_calc_margin(request["type"],SYMBOL,lots,entry)
    if margin is None or not math.isfinite(margin) or margin<0 or margin>acc.margin_free:
        raise RuntimeError("Insufficient or unknown free margin")
    logger.info("SIGNAL %s volume=%s entry~%s SL=%s TP=%s",token,lots,entry,sl,tp)
    if not live:
        state["actions"][token]={"status":"dry_run","request":request}
        save_state(state)
        return "dry_run"
    check=mt5.order_check(request)
    if check is None or check.retcode!=0:
        raise RuntimeError(f"Order preflight rejected: {check}")
    check_context(state,live)
    state["actions"][token]={"status":"pending","attempts":old.get("attempts",0)+1,
                             "started":time.time(),"request":request,"signal_id":review.signal_id if review else None,
                             "side":"long" if d==1 else "short","entry":entry,"stop":sl,"target":tp}
    save_state(state)  # must succeed BEFORE the network side effect
    result=mt5.order_send(request)
    status=submission_status(result)
    state["actions"][token].update(status=status,result=result_fields(result))
    save_state(state)
    logger.info("Submission %s status=%s result=%s",token,status,result)
    if status=="accepted":
        notifier.opened(side="long" if d==1 else "short",volume=getattr(result,"volume",None) or lots,
            entry=getattr(result,"price",None) or entry,stop=sl,target=tp,equity=acc.equity)
    return status


def close_reason(deal):
    """Readable reason for a closing deal (MT5 DEAL_REASON_*)."""
    names={getattr(mt5,"DEAL_REASON_SL",4):"stop_loss",getattr(mt5,"DEAL_REASON_TP",5):"take_profit",
           getattr(mt5,"DEAL_REASON_EXPERT",3):"bot",getattr(mt5,"DEAL_REASON_SO",6):"stop_out"}
    for manual in ("DEAL_REASON_CLIENT","DEAL_REASON_MOBILE","DEAL_REASON_WEB"):
        names[getattr(mt5,manual,{"DEAL_REASON_CLIENT":0,"DEAL_REASON_MOBILE":1,"DEAL_REASON_WEB":2}[manual])]="manual"
    return names.get(getattr(deal,"reason",None),"broker_close")


def reconcile(state,now):
    check_context(state,state["live"])
    if not state["live"]:
        return None
    since=pd.Timestamp(state["last_bar"]) if state["last_bar"] else now-pd.Timedelta(days=7)
    pending=[pd.Timestamp(v["started"],unit="s") for v in state["actions"].values() if v["status"] in ("pending","unknown")]
    # Positions whose close has not been seen yet: search from their entry, so a close
    # that happened while the bot was stopped (or before the fix above) is still reported.
    open_ours=[pd.Timestamp(v["started"],unit="s") for v in state["actions"].values()
               if v["status"]=="accepted" and v.get("position_id") and not v.get("closed") and v.get("started")]
    if pending or open_ours:
        since=min([since]+pending+open_ours)
    # Broker history is keyed by server wall-clock time, not UTC.
    deals=mt5.history_deals_get(utc_to_server_datetime(since-pd.Timedelta(minutes=15)),
                               utc_to_server_datetime(now+pd.Timedelta(minutes=1)))
    if deals is None:
        raise RuntimeError("Cannot reconcile broker deals")
    newest=None
    independent=hedging_account()
    exit_entries={getattr(mt5,"DEAL_ENTRY_OUT",1),getattr(mt5,"DEAL_ENTRY_OUT_BY",3)}
    for deal in deals:
        if deal.symbol!=SYMBOL:
            continue
        stamp=server_ms_to_utc(deal.time_msc)
        # A deal is ours by magic OR by position: a position closed by hand (terminal,
        # mobile, web) gets a closing deal with magic 0 -- before 2026-09-27 such closes
        # were ignored, so no Telegram message and no outcome report was sent.
        ours=deal.magic==MAGIC or any(a.get("position_id")==deal.position_id for a in state["actions"].values())
        # Our own exposure (or any exposure on netting) invalidates replay.
        if ours or not independent:
            newest=max(newest,stamp) if newest is not None else stamp
        for token,action in state["actions"].items():
            order=action.get("result",{}).get("order")
            if deal.magic==MAGIC and (deal.comment==token or (order and deal.order==order)):
                action["status"]="accepted"
                action["position_id"]=deal.position_id
                break
        if ours:
            state["deals"][str(deal.ticket)]={key:getattr(deal,key,None) for key in
                ("ticket","position_id","time_msc","entry","volume","price","profit","commission","swap","fee","reason")}
            if getattr(deal,"entry",None) in exit_entries and str(deal.ticket) not in state["outcomes"]:
                action=next((a for a in state["actions"].values()
                             if a.get("position_id")==deal.position_id),None)
                if action:
                    profit=float(deal.profit)+float(getattr(deal,"commission",0) or 0)+float(getattr(deal,"swap",0) or 0)
                    reason=close_reason(deal)
                    report_outcome(signal_id=action.get("signal_id"),profit=profit,exit_reason=reason,
                        ticket=deal.ticket,entry_price=action.get("entry"),close_price=deal.price,volume=deal.volume)
                    notifier.closed(side=action.get("side","unknown"),volume=deal.volume,entry=action.get("entry",0),
                        close=deal.price,profit=profit,reason=reason,equity=account().equity)
                    record_outcome(state,profit,stamp)
                    state["outcomes"][str(deal.ticket)]={"reported":True,"profit":profit}
                    action["closed"]=True
    save_state(state)
    return newest


def manage_positions(state,logger):
    if not state["live"]:
        return
    check_context(state,True)
    for pos in positions(SYMBOL):
        held=time.time()-server_seconds_to_utc_epoch(pos.time)
        if pos.magic!=MAGIC or (pos.sl>0 and held<DEFAULTS.max_hold_minutes*60):
            continue
        key=str(pos.ticket)
        previous=state["closes"].get(key)
        if previous and previous["status"] in ("pending","unknown","accepted") and pos.volume>=previous["volume"]-1e-10:
            logger.error("Close %s awaits broker reconciliation",key)
            continue
        tick,info=get_tick(SYMBOL),get_symbol_info(SYMBOL)
        orders=mt5.orders_get(symbol=SYMBOL)
        if orders is None or any(o.position_id==pos.ticket for o in orders):
            continue
        d=-1 if pos.type==mt5.ORDER_TYPE_BUY else 1
        request=dict(action=mt5.TRADE_ACTION_DEAL,symbol=SYMBOL,volume=pos.volume,
            type=mt5.ORDER_TYPE_BUY if d==1 else mt5.ORDER_TYPE_SELL,position=pos.ticket,
            deviation=DEVIATION_POINTS,magic=MAGIC,comment="slp2:time_exit",
            type_time=mt5.ORDER_TIME_GTC,type_filling=filling_mode(info))
        if info.trade_exemode!=mt5.SYMBOL_TRADE_EXECUTION_MARKET:
            request["price"]=tick.ask if d==1 else tick.bid
        check=mt5.order_check(request)
        if check is None or check.retcode!=0:
            raise RuntimeError(f"Close preflight failed: {check}")
        check_context(state,True)
        state["closes"][key]={"status":"pending","volume":pos.volume}
        save_state(state)
        result=mt5.order_send(request)
        state["closes"][key].update(status=submission_status(result),result=result_fields(result))
        save_state(state)
        logger.info("Time/protection exit %s result=%s",key,result)


def lock_breakeven(state,df,logger):
    """Move our own stop to entry after a closed bar has reached BREAKEVEN_AT_R.

    Only ever tightens; SLTP is idempotent, so a failed or uncertain attempt is
    simply retried on a later poll while the original stop keeps protecting.
    """
    if not state["live"] or BREAKEVEN_AT_R is None:
        return
    check_context(state,True)
    for pos in positions(SYMBOL):
        if pos.magic!=MAGIC:
            continue
        position_id=getattr(pos,"identifier",pos.ticket)
        action=next((a for a in state["actions"].values() if a.get("position_id")==position_id),None)
        if action is None or not action.get("stop"):
            continue  # not reconciled yet; the original stop still applies
        d=1 if pos.type==mt5.ORDER_TYPE_BUY else -1
        entry=pos.price_open
        distance=d*(entry-action["stop"])
        if distance<=0 or (pos.sl>0 and d*(pos.sl-entry)>=0):
            continue  # already at or beyond breakeven
        opened=server_ms_to_utc(getattr(pos,"time_msc",pos.time*1000))
        bars=df[df.bar_time>=opened.floor("15min")]
        if bars.empty:
            continue
        tick,info=get_tick(SYMBOL),get_symbol_info(SYMBOL)
        best=bars.high.max() if d==1 else bars.low.min()+(tick.ask-tick.bid)
        if d*(best-entry)/distance<BREAKEVEN_AT_R:
            continue
        new_sl=quantize_price(entry,info,up=d==1)
        reference=tick.bid if d==1 else tick.ask
        if d*(reference-new_sl)<max(info.trade_stops_level*info.point,info.trade_tick_size):
            logger.info("Breakeven for %s postponed: price too close to entry",pos.ticket)
            continue
        request=dict(action=getattr(mt5,"TRADE_ACTION_SLTP",6),symbol=SYMBOL,position=pos.ticket,
                     sl=new_sl,tp=pos.tp,magic=MAGIC)
        check=mt5.order_check(request)
        if check is None or check.retcode!=0:
            logger.warning("Breakeven preflight for %s rejected: %s",pos.ticket,check)
            continue
        check_context(state,True)
        result=mt5.order_send(request)
        state["locks"][str(pos.ticket)]={"sl":new_sl,"status":submission_status(result),"result":result_fields(result)}
        save_state(state)
        logger.info("Breakeven stop %s -> %s result=%s",pos.ticket,new_sl,result)


def update_pattern(df,state,*,blocked=False,activity=None):
    """Only advance newly closed bars, resetting after observed broker exposure.

    Fresh starts and missed history start from the current bar without inventing
    an earlier account path. Historical catch-up may build pending setups but
    cannot turn an old trigger into a current entry.
    """
    latest=pd.Timestamp(df.bar_time.iloc[-1])
    if state["last_bar"] is None or pd.Timestamp(state["last_bar"])<pd.Timestamp(df.bar_time.iloc[0]):
        state.update(last_bar=str(latest),pending=None,candidate=None)
        return
    if pd.Timestamp(state["last_bar"])>latest:
        raise RuntimeError("History moved backwards; retain checkpoint")
    if blocked:
        state.update(last_bar=str(latest),pending=None,candidate=None)
        return
    if activity is not None and activity>=pd.Timestamp(state["last_bar"]):
        cutoff=pd.Timestamp(activity).floor("15min")
        state.update(last_bar=str(min(latest,cutoff)),pending=None,candidate=None)
    engine=PatternEngine(df,strategy_parameters())
    for i in range(len(df)):
        stamp=pd.Timestamp(engine.times[i])
        if stamp<=pd.Timestamp(state["last_bar"]):
            continue
        state["pending"],trigger=engine.advance(state["pending"],i)
        state["last_bar"]=str(stamp)
        if trigger is not None:
            state["candidate"]=trigger if stamp==latest else None


def setup_logger():
    RUNTIME_DIR.mkdir(parents=True,exist_ok=True)
    logger=logging.getLogger("slp2")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        for handler in (RotatingFileHandler(LOG_PATH,maxBytes=5000000,backupCount=3,encoding="utf-8"),logging.StreamHandler()):
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            logger.addHandler(handler)
    return logger


def run(live):
    validate_risk(RISK_PCT)
    logger=setup_logger()
    handle=acquire_lock()
    try:
        if not mt5.initialize():
            raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
        bound=identity()
        if not mt5.symbol_select(SYMBOL,True):
            raise RuntimeError(f"Symbol unavailable: {SYMBOL}")
        state=load_state(bound,live)
        logger.info("Starting mode=%s account=%s config=%s",live,bound,state["config"])
        last_error_notice=last_clock_notice=0.0
        while True:
            try:
                check_context(state,live)
                if not refresh_server_offset(SYMBOL):
                    # Market closed/stale feed at startup: no UTC mapping yet, so no bars or orders.
                    if time.time()-last_clock_notice>=900:
                        logger.info("Waiting for a fresh %s tick to learn the broker server offset",SYMBOL)
                        last_clock_notice=time.time()
                    time.sleep(POLL_SECONDS)
                    continue
                manage_positions(state,logger)
                now=utc_now()
                activity=reconcile(state,now)
                df=get_bars(SYMBOL,"M15",BARS_NEEDED,asof=now)
                lock_breakeven(state,df,logger)
                update_pattern(df,state,blocked=has_open_position(logger),activity=activity)
                save_state(state)
                if state["candidate"] is not None:
                    status=place_order(state["candidate"],live,logger,state)
                    if status in ("accepted","dry_run","consumed","skipped","ai_rejected","guard_blocked"):
                        state["candidate"]=None
                        save_state(state)
            except Exception as exc:
                logger.exception("Poll failed; retained checkpoint, no blind retry")
                if time.time()-last_error_notice>=900:
                    notifier.error(str(exc))
                    last_error_notice=time.time()
            time.sleep(POLL_SECONDS)
    except KeyboardInterrupt:
        logger.info("Stopped by user")
    finally:
        mt5.shutdown()
        handle.close()


def main():
    parser=argparse.ArgumentParser(description="SLP2 M15: dry-run by default")
    parser.add_argument("--live",action="store_true",help="Submit real orders")
    args=parser.parse_args()
    run(args.live)


if __name__=="__main__":
    main()
