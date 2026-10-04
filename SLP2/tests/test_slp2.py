"""Offline regression tests. No test can connect to an MT5 terminal."""
from pathlib import Path
import contextlib
import json
import logging
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import Mock, patch
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
mt5=types.ModuleType("MetaTrader5")
for k,v in {"TIMEFRAME_M5":5,"TIMEFRAME_M15":15,"TIMEFRAME_H1":60,"ORDER_TYPE_BUY":0,
    "ORDER_TYPE_SELL":1,"ORDER_FILLING_FOK":0,"ORDER_FILLING_IOC":1,"ORDER_FILLING_RETURN":2,
    "SYMBOL_TRADE_EXECUTION_MARKET":2,"TRADE_ACTION_DEAL":1,"ORDER_TIME_GTC":0,
    "TRADE_RETCODE_DONE":10009,"TRADE_RETCODE_DONE_PARTIAL":10010}.items():
    setattr(mt5,k,v)
for name in ("initialize","shutdown","symbol_select","account_info","symbol_info","symbol_info_tick",
             "positions_get","orders_get","order_calc_profit","order_calc_margin","order_check",
             "order_send","history_deals_get","copy_rates_from_pos","copy_rates_range"):
    setattr(mt5,name,Mock(side_effect=AssertionError("Unconfigured offline API "+name)))
mt5.last_error=lambda:(-1,"offline")
sys.modules["MetaTrader5"]=mt5
import SLP2 as live
from bot import risk, mt5_data
from bot.sp2l import PatternEngine, Parameters, add_indicators, detect_buy_setup
from bot.state import atomic_json
from scripts.sp2l_m15_backtest import simulate,summarize
from scripts import pattern_strategy as pattern


def setup_frame():
    # A four-candle spike/gap setup, closed-bar pullback, then obtainable entry.
    rows=[(100,101.2,99.5,101),(101,105.2,100.8,105),
          (105,106.3,104.5,106),(105.8,106.4,103.5,105.2),
          (104,105.5,102.5,104.5),(104.8,105.5,104,105)]
    frame=pd.DataFrame(rows,columns=["open","high","low","close"])
    frame["bar_time"]=pd.date_range("2026-01-01",periods=len(frame),freq="15min")
    frame["avg_spread_price"]=.4
    return frame


class LiveTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.stack=contextlib.ExitStack()
        self.addCleanup(self.tmp.cleanup)
        mt5_data.reset_server_clock(0)  # the fake API below stamps in UTC
        self.addCleanup(mt5_data.reset_server_clock)
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(live,"RUNTIME_DIR",Path(self.tmp.name)))
        self.stack.enter_context(patch.object(live,"LOCK_PATH",Path(self.tmp.name)/"lock"))
        self.stack.enter_context(patch.object(live,"review_signal",return_value=types.SimpleNamespace(
            approved=True,decision="APPROVE",reason="offline test",signal_id="offline-signal")))
        self.stack.enter_context(patch.object(live,"log_agent_detail"))
        self.stack.enter_context(patch.object(live.notifier,"opened"))
        self.stack.enter_context(patch.object(live.notifier,"closed"))
        self.stack.enter_context(patch.object(live.notifier,"error"))
        self.now=pd.Timestamp("2026-09-23 10:16")
        self.stack.enter_context(patch.object(live,"utc_now",return_value=self.now))
        self.acc=types.SimpleNamespace(server="offline",login=1,currency="USD",equity=25000,balance=25000,margin_free=20000,margin_mode=0)
        self.info=types.SimpleNamespace(trade_tick_size=.01,point=.01,volume_min=.01,volume_max=100.,
            volume_step=.01,trade_stops_level=0,filling_mode=1,trade_exemode=2)
        self.tick=types.SimpleNamespace(bid=105.,ask=105.2,time_msc=int(time.time()*1000))
        values={"account_info":self.acc,"symbol_info":self.info,"symbol_info_tick":self.tick,
                "positions_get":(),"orders_get":(),"order_check":types.SimpleNamespace(retcode=0),
                "order_calc_margin":10.,"history_deals_get":(),"shutdown":None}
        for name,value in values.items():
            self.stack.enter_context(patch.object(mt5,name,return_value=value))
        self.stack.enter_context(patch.object(mt5,"order_calc_profit",side_effect=lambda typ,s,v,a,b:(1 if typ==0 else -1)*(b-a)*100*v))
        self.state=live.load_state("offline:1:USD",True)
        self.trigger=dict(direction=1,bar_time="2026-09-23 10:00",decision_time="2026-09-23 10:15",stop=99.5,ref_price=104.)
        self.logger=logging.getLogger("offline-test")
        self.logger.addHandler(logging.NullHandler())

    def test_actual_entry_sets_target_and_budget_after_quantization(self):
        self.info.trade_tick_size=.25
        self.trigger["stop"]=99.61
        result=types.SimpleNamespace(retcode=10009,order=1,deal=2,price=105.2,volume=.08)
        with patch.object(mt5,"order_send",return_value=result) as send:
            self.assertEqual(live.place_order(self.trigger,True,self.logger,self.state),"accepted")
        req=send.call_args.args[0]
        self.assertEqual(req["sl"],99.5)
        self.assertAlmostEqual(req["tp"],133.75)  # 105.2+5*(105.2-99.5), rounded up to the .25 tick
        loss_per_lot=(105.2+.2-99.5)*100+7
        self.assertLessEqual(req["volume"]*loss_per_lot,self.acc.equity*live.RISK_PCT)  # the risk budget, whatever RISK_PCT is

    def test_actual_stop_cannot_exceed_strategy_limit(self):
        self.trigger["stop"]=90.
        with patch.object(mt5,"order_send") as send:
            self.assertEqual(live.place_order(self.trigger,True,self.logger,self.state),"skipped")
        send.assert_not_called()

    def test_stop_not_wider_than_three_spreads_is_skipped(self):
        # spread 0.2 -> limit 0.6; entry 105.2: stop 104.7 is 0.5 away (skipped), stop 104.5 is 0.7 away (allowed)
        self.trigger["stop"]=104.7
        with patch.object(mt5,"order_send") as send:
            self.assertEqual(live.place_order(self.trigger,True,self.logger,self.state),"skipped")
        send.assert_not_called()
        self.assertTrue(live.stop_too_tight(.6,self.tick))
        self.assertFalse(live.stop_too_tight(.7,self.tick))

    def test_ai_rejection_blocks_order_and_is_journaled(self):
        denied=types.SimpleNamespace(approved=False,decision="REJECT",reason="test rejection",signal_id="ai-1")
        with patch.object(live,"review_signal",return_value=denied),patch.object(mt5,"order_send") as send:
            self.assertEqual(live.place_order(self.trigger,True,self.logger,self.state),"ai_rejected")
        send.assert_not_called()
        action=next(iter(self.state["actions"].values()))
        self.assertEqual(action["status"],"ai_rejected")
        self.assertEqual(action["signal_id"],"ai-1")

    def test_cooldown_blocks_order_without_calling_ai_or_broker(self):
        self.state["guards"]["day"]=str(self.now.date())
        self.state["guards"]["day_start_equity"]=self.acc.equity
        self.state["guards"]["cooldown_until"]=str(self.now+pd.Timedelta(hours=1))
        with patch.object(live,"review_signal") as review,patch.object(mt5,"order_send") as send:
            self.assertEqual(live.place_order(self.trigger,True,self.logger,self.state),"guard_blocked")
        review.assert_not_called()
        send.assert_not_called()

    def test_age_is_measured_from_close_not_open(self):
        dry=live.load_state("offline:1:USD",False)
        with patch.object(mt5,"order_send") as send:
            self.assertEqual(live.place_order(self.trigger,False,self.logger,dry),"dry_run")
        send.assert_not_called()

    def test_old_or_future_signal_is_rejected(self):
        for decision in ("2026-09-23 09:00","2026-09-23 11:00"):
            trigger=self.trigger | {"decision_time":decision}
            with patch.object(mt5,"order_send") as send:
                self.assertEqual(live.place_order(trigger,True,self.logger,self.state),"skipped")
            send.assert_not_called()

    def test_manual_position_blocks_netting_entry(self):
        with patch.object(mt5,"positions_get",return_value=[types.SimpleNamespace(magic=1)]),patch.object(mt5,"order_send") as send:
            self.assertEqual(live.place_order(self.trigger,True,self.logger,self.state),"skipped")
        send.assert_not_called()

    def test_other_ea_position_does_not_block_on_hedging(self):
        self.acc.margin_mode=2
        other=[types.SimpleNamespace(magic=991015)]
        with patch.object(mt5,"positions_get",return_value=other),patch.object(mt5,"orders_get",return_value=other),             patch.object(mt5,"order_send",return_value=types.SimpleNamespace(retcode=10009,order=1,deal=1,price=105.2,volume=.1)) as send:
            self.assertEqual(live.place_order(self.trigger,True,self.logger,self.state),"accepted")
        send.assert_called_once()

    def test_own_position_still_blocks_on_hedging(self):
        self.acc.margin_mode=2
        with patch.object(mt5,"positions_get",return_value=[types.SimpleNamespace(magic=live.MAGIC)]),patch.object(mt5,"order_send") as send:
            self.assertEqual(live.place_order(self.trigger,True,self.logger,self.state),"skipped")
        send.assert_not_called()

    def test_other_ea_deal_does_not_reset_pattern_on_hedging(self):
        self.acc.margin_mode=2
        deal=types.SimpleNamespace(symbol=live.SYMBOL,time_msc=int(self.now.timestamp()*1000),magic=991015,
            comment="",order=9,ticket=9,position_id=9,profit=0,commission=0,swap=0,fee=0,entry=0)
        with patch.object(mt5,"history_deals_get",return_value=[deal]):
            self.assertIsNone(live.reconcile(self.state,self.now))
        self.acc.margin_mode=0
        with patch.object(mt5,"history_deals_get",return_value=[deal]):
            self.assertIsNotNone(live.reconcile(self.state,self.now))

    def _lock_case(self,high,sl=99.5,magic=None,stops_level=0,enabled=2.0):
        """Long filled at 105.2 with stop 99.5 (1R=5.7, 2R reached at 116.6)."""
        self.state["actions"]["slp2:x"]={"status":"accepted","position_id":7,"stop":99.5}
        pos=types.SimpleNamespace(magic=live.MAGIC if magic is None else magic,ticket=7,identifier=7,type=0,
            price_open=105.2,sl=sl,tp=133.7,volume=.1,time=int(pd.Timestamp("2026-09-23 10:02").timestamp()),
            time_msc=int(pd.Timestamp("2026-09-23 10:02").timestamp()*1000))
        bars=pd.DataFrame({"bar_time":pd.to_datetime(["2026-09-23 09:45","2026-09-23 10:00"]),
            "open":[105.,105.2],"high":[200.,high],"low":[104.,105.],"close":[105.,110.]})  # 09:45 precedes entry
        self.tick.bid,self.tick.ask=112.,112.2
        self.info.trade_stops_level=stops_level
        with patch.object(mt5,"positions_get",return_value=[pos]),patch.object(live,"BREAKEVEN_AT_R",enabled),             patch.object(mt5,"order_send",return_value=types.SimpleNamespace(retcode=10009)) as send:
            live.lock_breakeven(self.state,bars,self.logger)
        return send

    def test_breakeven_is_off_when_disabled(self):
        self._lock_case(high=120.,enabled=None).assert_not_called()

    def test_breakeven_moves_stop_after_closed_bar_reaches_2r(self):
        send=self._lock_case(high=116.7)
        req=send.call_args.args[0]
        self.assertEqual((req["position"],req["sl"],req["tp"]),(7,105.2,133.7))

    def test_breakeven_ignores_bars_before_entry_and_below_2r(self):
        self._lock_case(high=116.).assert_not_called()

    def test_breakeven_never_loosens_or_touches_other_eas(self):
        self._lock_case(high=120.,sl=106.).assert_not_called()
        self._lock_case(high=120.,magic=991015).assert_not_called()

    def test_breakeven_waits_when_broker_stop_level_forbids_it(self):
        self._lock_case(high=120.,stops_level=1000).assert_not_called()

    def test_unknown_position_or_order_state_blocks_entry(self):
        for name in ("positions_get","orders_get"):
            with patch.object(mt5,name,return_value=None),patch.object(mt5,"order_send") as send,self.assertRaises(RuntimeError):
                live.place_order(self.trigger,True,self.logger,self.state)
            send.assert_not_called()

    def test_margin_is_checked(self):
        with patch.object(mt5,"order_calc_margin",return_value=999999),patch.object(mt5,"order_send") as send,self.assertRaises(RuntimeError):
            live.place_order(self.trigger,True,self.logger,self.state)
        send.assert_not_called()

    def test_account_switch_blocks_order(self):
        self.acc.login=2
        with patch.object(mt5,"order_send") as send,self.assertRaises(RuntimeError):
            live.place_order(self.trigger,True,self.logger,self.state)
        send.assert_not_called()

    def test_dry_state_cannot_be_used_live(self):
        dry=live.load_state("offline:1:USD",False)
        with self.assertRaises(RuntimeError):
            live.place_order(self.trigger,True,self.logger,dry)

    def test_dry_run_does_not_consume_live_signal(self):
        dry=live.load_state("offline:1:USD",False)
        live.place_order(self.trigger,False,self.logger,dry)
        result=types.SimpleNamespace(retcode=10009)
        with patch.object(mt5,"order_send",return_value=result) as send:
            live.place_order(self.trigger,True,self.logger,self.state)
            self.assertEqual(send.call_count,1)

    def test_crash_during_send_persists_intent(self):
        with patch.object(mt5,"order_send",side_effect=RuntimeError("disconnect")),self.assertRaises(RuntimeError):
            live.place_order(self.trigger,True,self.logger,self.state)
        restarted=live.load_state("offline:1:USD",True)
        self.assertTrue(live.unresolved(restarted))
        with patch.object(mt5,"order_send") as send,self.assertRaises(RuntimeError):
            live.place_order(self.trigger,True,self.logger,restarted)
        send.assert_not_called()

    def test_partial_fill_is_accepted_and_never_duplicated(self):
        result=types.SimpleNamespace(retcode=10010,volume=.01,price=105.3,order=1,deal=2)
        with patch.object(mt5,"order_send",return_value=result) as send:
            self.assertEqual(live.place_order(self.trigger,True,self.logger,self.state),"accepted")
            restarted=live.load_state("offline:1:USD",True)
            self.assertEqual(live.place_order(self.trigger,True,self.logger,restarted),"consumed")
            self.assertEqual(send.call_count,1)

    def test_definite_rejections_have_bounded_retry(self):
        with patch.object(mt5,"order_send",return_value=types.SimpleNamespace(retcode=10004)) as send:
            for _ in range(5):
                live.place_order(self.trigger,True,self.logger,self.state)
            self.assertEqual(send.call_count,3)

    def test_failed_checkpoint_never_sends(self):
        with patch.object(live,"save_state",side_effect=OSError("disk full")),patch.object(mt5,"order_send") as send,self.assertRaises(OSError):
            live.place_order(self.trigger,True,self.logger,self.state)
        send.assert_not_called()

    def test_corrupt_state_does_not_reset_guard(self):
        live.state_path("offline:1:USD",True).write_text("{")
        with self.assertRaises(json.JSONDecodeError):
            live.load_state("offline:1:USD",True)

    def test_lock_always_targets_first_byte(self):
        live.LOCK_PATH.write_bytes(b"1234567")
        with live.acquire_lock():
            with self.assertRaises(OSError):
                live.acquire_lock()
        with live.acquire_lock():
            pass

    def test_reconcile_recovers_unknown_submission(self):
        with patch.object(mt5,"order_send",return_value=None):
            live.place_order(self.trigger,True,self.logger,self.state)
        token=next(iter(self.state["actions"]))
        deal=types.SimpleNamespace(symbol=live.SYMBOL,time_msc=int(self.now.timestamp()*1000),magic=live.MAGIC,
            comment=token,order=4,ticket=5,position_id=6,profit=0,commission=-3.5,swap=0,fee=0)
        with patch.object(mt5,"history_deals_get",return_value=[deal]):
            live.reconcile(self.state,self.now)
            live.reconcile(self.state,self.now)
        self.assertFalse(live.unresolved(self.state))
        self.assertEqual(len(self.state["deals"]),1)

    def _closing(self,magic,reason,position_id=6):
        self.state["actions"]["slp2:x"]={"status":"accepted","position_id":6,"side":"short","entry":4279.8,"signal_id":"sig"}
        deal=types.SimpleNamespace(symbol=live.SYMBOL,time_msc=int(self.now.timestamp()*1000),magic=magic,comment="",order=9,
            ticket=77,position_id=position_id,entry=1,volume=.13,price=4275.0,profit=62.4,commission=-.9,swap=0,fee=0,reason=reason)
        with patch.object(mt5,"history_deals_get",return_value=[deal]),patch.object(live,"report_outcome") as rep:
            live.reconcile(self.state,self.now); live.reconcile(self.state,self.now)
        return rep

    def test_manual_close_with_magic_zero_is_notified_once(self):
        rep=self._closing(magic=0,reason=1)                      # closed from the mobile app
        live.notifier.closed.assert_called_once()
        self.assertEqual(live.notifier.closed.call_args.kwargs["reason"],"manual")
        self.assertAlmostEqual(live.notifier.closed.call_args.kwargs["profit"],61.5)
        self.assertEqual(rep.call_args.kwargs["exit_reason"],"manual")

    def test_stop_loss_close_names_the_reason(self):
        self._closing(magic=live.MAGIC,reason=4)
        self.assertEqual(live.notifier.closed.call_args.kwargs["reason"],"stop_loss")

    def test_close_before_last_bar_is_found_from_entry_time(self):
        self.state["last_bar"]="2026-09-25 20:45:00"                     # bot stopped long after the close
        self.state["actions"]["slp2:y"]={"status":"accepted","position_id":8,"side":"short","entry":4279.8,
                                         "started":pd.Timestamp("2026-09-25 15:00:26").timestamp()}
        with patch.object(mt5,"history_deals_get",return_value=[]) as q:
            live.reconcile(self.state,self.now)
        self.assertLessEqual(q.call_args.args[0].replace(tzinfo=None),pd.Timestamp("2026-09-25 14:45:26").to_pydatetime())
        deal=types.SimpleNamespace(symbol=live.SYMBOL,time_msc=int(pd.Timestamp("2026-09-25 15:20").timestamp()*1000),magic=0,
            comment="",order=1,ticket=91,position_id=8,entry=1,volume=.13,price=4283.5,profit=-48.,commission=0,swap=0,fee=0,reason=1)
        with patch.object(mt5,"history_deals_get",return_value=[deal]),patch.object(live,"report_outcome"):
            live.reconcile(self.state,self.now)
        live.notifier.closed.assert_called_once()
        self.assertTrue(self.state["actions"]["slp2:y"]["closed"])

    def test_foreign_manual_trade_is_not_notified(self):
        self._closing(magic=0,reason=1,position_id=999)          # someone else's position
        live.notifier.closed.assert_not_called()

    def test_uncertain_time_exit_not_resent(self):
        pos=types.SimpleNamespace(magic=live.MAGIC,sl=100,time=0,ticket=3,volume=.1,type=0)
        with patch.object(mt5,"positions_get",return_value=[pos]),patch.object(mt5,"order_send",return_value=None) as send:
            live.manage_positions(self.state,self.logger)
            live.manage_positions(self.state,self.logger)
            self.assertEqual(send.call_count,1)

    def test_startup_does_not_replay_fictitious_trades(self):
        frame=setup_frame()
        live.update_pattern(frame,self.state)
        self.assertIsNone(self.state["candidate"])
        self.assertIsNone(self.state["pending"])
        self.assertEqual(pd.Timestamp(self.state["last_bar"]),frame.bar_time.iloc[-1])

    def test_position_clears_pending_and_skips_occupied_bars(self):
        frame=setup_frame()
        self.state.update(last_bar=str(frame.bar_time.iloc[2]),pending={"direction":1})
        live.update_pattern(frame,self.state,blocked=True)
        self.assertIsNone(self.state["pending"])
        self.assertEqual(pd.Timestamp(self.state["last_bar"]),frame.bar_time.iloc[-1])

    def test_incremental_pattern_matches_shared_flat_scan(self):
        frame=setup_frame().iloc[:5]
        self.state["last_bar"]=str(frame.bar_time.iloc[2])
        with patch.object(live,"USE_EMA_FILTER",False):
            live.update_pattern(frame.iloc[:4],self.state)
            self.assertIsNotNone(self.state["pending"])
            live.update_pattern(frame,self.state)
            _,trigger=live.latest_pending_and_trigger(frame)
        self.assertEqual(self.state["candidate"],trigger)


class ResearchTests(unittest.TestCase):
    def test_fourth_candle_can_form_setup(self):
        frame=setup_frame()
        arrays=[frame[c].to_numpy() for c in ("open","high","low","close")]
        self.assertTrue(detect_buy_setup(*arrays,3,2))

    def test_market_entry_is_next_open_not_trigger_extreme(self):
        frame=setup_frame()
        trades=simulate(frame,use_ema_filter=False,slippage_points=0)
        self.assertEqual(len(trades),1)
        row=trades.iloc[0]
        self.assertAlmostEqual(row.entry_price,105.2)
        self.assertEqual(row.entry_time,frame.bar_time.iloc[5])
        self.assertAlmostEqual(row.target_price,105.2+Parameters().rr*(105.2-99.5))

    def test_profit_lock_moves_stop_only_after_bar_close(self):
        frame=setup_frame()  # long at 105.2, stop 99.5 (1R = 5.7)
        extra=pd.DataFrame([(105.3,117.,105.1,116.),(116.,116.5,100.,101.)],columns=["open","high","low","close"])
        extra["bar_time"]=frame.bar_time.iloc[-1]+pd.to_timedelta([15,30],unit="min")
        extra["avg_spread_price"]=.4
        frame=pd.concat([frame,extra],ignore_index=True)
        plain=simulate(frame,use_ema_filter=False,slippage_points=0)
        locked=simulate(frame,use_ema_filter=False,slippage_points=0,lock_steps=((2.,0.),))
        self.assertEqual(plain.exit_reason.iloc[0],"end_of_data")
        self.assertEqual(locked.exit_reason.iloc[0],"locked_stop")
        self.assertAlmostEqual(locked.exit_price.iloc[0],105.2)
        # Reaching the level intrabar on the SAME bar that reverses must not lock.
        same=frame.copy(); same.loc[6,["high","low","close"]]=[117.,99.,100.]
        self.assertEqual(simulate(same,use_ema_filter=False,slippage_points=0,lock_steps=((2.,0.),)).exit_reason.iloc[0],"stop")

    def test_early_exit_uses_next_open_after_deadline(self):
        frame=setup_frame()  # long at 105.2, stop 99.5 (1R = 5.7)
        extra=pd.DataFrame([(105.3,106.,104.5,105.5),(105.5,106.,104.8,105.2),(104.9,105.5,104.,105.)],
                           columns=["open","high","low","close"])
        extra["bar_time"]=frame.bar_time.iloc[-1]+pd.to_timedelta([15,30,45],unit="min")
        extra["avg_spread_price"]=.4
        frame=pd.concat([frame,extra],ignore_index=True)
        t=simulate(frame,use_ema_filter=False,slippage_points=0,early_exit=(.5,1.))
        self.assertEqual(t.exit_reason.iloc[0],"early_exit")
        self.assertEqual(t.exit_time.iloc[0],extra.bar_time.iloc[1])  # 30 min after entry -> next open
        self.assertAlmostEqual(t.exit_price.iloc[0],105.5)

    def test_weekend_close_exits_friday_and_blocks_late_entries(self):
        frame=setup_frame()
        frame["bar_time"]=pd.date_range("2026-01-02 21:30",periods=len(frame),freq="15min")  # a Friday
        extra=pd.DataFrame([(105.3,106.,104.5,105.5),(105.5,106.,104.8,105.2)],columns=["open","high","low","close"])
        extra["bar_time"]=frame.bar_time.iloc[-1]+pd.to_timedelta([15,30],unit="min")
        extra["avg_spread_price"]=.4
        frame=pd.concat([frame,extra],ignore_index=True)  # entry bar 22:45, next bars 23:00, 23:15
        t=simulate(frame,use_ema_filter=False,slippage_points=0,weekend_close_hour=23)
        self.assertEqual((t.exit_reason.iloc[0],t.exit_time.iloc[0]),("weekend",pd.Timestamp("2026-01-02 23:00")))
        self.assertTrue(simulate(frame,use_ema_filter=False,slippage_points=0,weekend_close_hour=22).empty)

    def test_open_end_of_data_trade_is_recorded(self):
        trades=simulate(setup_frame(),use_ema_filter=False)
        self.assertEqual(trades.iloc[0].exit_reason,"end_of_data")

    def test_time_exit_is_recorded(self):
        frame=setup_frame()
        frame.loc[6]=frame.iloc[-1]
        frame.loc[6,"bar_time"]=frame.bar_time.iloc[5]+pd.Timedelta(minutes=15)
        trades=simulate(frame,use_ema_filter=False,max_hold_bars=1)
        self.assertEqual(trades.iloc[0].exit_reason,"time")

    def test_gap_stop_uses_opening_quote(self):
        frame=setup_frame()
        frame.loc[6]=frame.iloc[-1]
        frame.loc[6,"bar_time"]=frame.bar_time.iloc[5]+pd.Timedelta(minutes=15)
        frame.loc[6,["open","high","low","close"]]=[90.,92.,89.,91.]
        trades=simulate(frame,use_ema_filter=False,slippage_points=0)
        self.assertEqual(trades.iloc[0].exit_price,90)
        self.assertEqual(trades.iloc[0].exit_reason,"stop_gap")

    def test_entry_bar_stop_is_not_ignored(self):
        frame=setup_frame()
        frame.loc[5,"low"]=98
        trades=simulate(frame,use_ema_filter=False,slippage_points=0)
        self.assertEqual(trades.iloc[0].exit_reason,"stop")

    def test_bid_ask_short_entry_and_exit(self):
        frame=setup_frame()
        original=frame.copy()
        frame["open"]=220-original.open
        frame["close"]=220-original.close
        frame["high"]=220-original.low
        frame["low"]=220-original.high
        trades=simulate(frame,use_ema_filter=False,slippage_points=0)
        self.assertEqual(len(trades),1)
        self.assertAlmostEqual(trades.iloc[0].entry_price,115.2)
        self.assertAlmostEqual(trades.iloc[0].exit_price,115.4)

    def test_missing_spread_fails_costed_simulation(self):
        with self.assertRaises(ValueError):
            simulate(setup_frame().drop(columns="avg_spread_price"))

    def test_entry_rejected_when_stop_already_beyond_exit_quote(self):
        frame=setup_frame()
        frame.loc[5,["open","high","low","close"]]=[99.4,100.,99.,99.6]
        self.assertTrue(simulate(frame,use_ema_filter=False).empty)

    def test_empty_results_have_schema(self):
        trades=simulate(setup_frame().iloc[:3],use_ema_filter=False)
        self.assertEqual(summarize(trades)["n_trades"],0)
        self.assertIn("entry_time",trades)

    def test_ema_tail_and_full_history_match(self):
        n=1000
        closes=100+np.sin(np.arange(n)/7)
        frame=pd.DataFrame({"bar_time":pd.date_range("2026-01-01",periods=n,freq="15min"),
            "open":closes,"close":closes,"high":closes+1,"low":closes-1})
        a,b=add_indicators(frame),add_indicators(frame.tail(250))
        self.assertAlmostEqual(a.EMA.iloc[-1],b.EMA.iloc[-1],places=12)

    def test_future_candles_do_not_change_past_trigger(self):
        frame=setup_frame()
        p=Parameters(ema_filter=False)
        a,b=PatternEngine(frame.iloc[:5],p),PatternEngine(frame,p)
        pa=pb=None
        for i in range(5):
            pa,ta=a.advance(pa,i)
            pb,tb=b.advance(pb,i)
            self.assertEqual((pa,ta),(pb,tb))

    def test_drawdown_starts_at_zero(self):
        trades=pd.DataFrame({"exit_time":pd.date_range("2026-01-01",periods=2),"r_multiple":[-1.,-1.]})
        self.assertEqual(summarize(trades)["max_dd_r"],2)

    def test_pattern_negative_index_cannot_read_future_tail(self):
        frame=setup_frame().assign(atr=1.)
        self.assertFalse(pattern.three_candle_pattern(frame,0,1))

    def test_zero_size_fvg_is_not_a_gap(self):
        frame=pd.DataFrame({"bar_time":pd.date_range("2026-01-01",periods=3,freq="15min"),
            "open":[100,101,102],"close":[101,102,103],"low":[99,100,101.5],"high":[101.5,102.5,103.5],"atr":[1.,1.,1.]})
        self.assertFalse(pattern.three_candle_pattern(frame,2,1,min_gap_mult=0,require_fvg=True))

    def test_h1_merge_preserves_unsorted_input_alignment(self):
        h1=pd.DataFrame({"bar_time":pd.date_range("2026-01-01",periods=8,freq="h"),
            "open":[100]*8,"close":[100]*8,"high":[101,102,103,102,101,102,104,102],"low":[99,98,97,98,99,98,96,98]})
        m15=h1.iloc[[6,3,5]].copy()
        result=pattern.attach_h1_trend(m15,h1.iloc[::-1],legs=1)
        sorted_result=pattern.attach_h1_trend(m15.sort_values("bar_time"),h1,legs=1)
        pd.testing.assert_series_equal(result.sort_values("bar_time").h1_trend,sorted_result.h1_trend)

    def test_invalid_h1_mode_cannot_silently_disable_filter(self):
        with self.assertRaises(ValueError):
            pattern.generate_signals(setup_frame(),h1_mode="typo")


class ServerClockTests(unittest.TestCase):
    """MT5 stamps ticks/bars/deals/positions in broker server time (here UTC+3)."""
    OFFSET=3*3600

    def setUp(self):
        mt5_data.reset_server_clock()
        self.addCleanup(mt5_data.reset_server_clock)
        self.now=time.time()

    def tick(self,age=0.0,offset=OFFSET):
        return types.SimpleNamespace(bid=105.,ask=105.2,time_msc=int((self.now+offset-age)*1000))

    def test_fresh_server_time_tick_is_accepted_and_offset_learned(self):
        with patch.object(mt5,"symbol_info_tick",return_value=self.tick(age=1)):
            mt5_data.get_tick("XAUUSD",now=self.now)
        self.assertEqual(mt5_data.server_offset_seconds(),self.OFFSET)

    def test_stale_server_time_tick_is_still_rejected(self):
        for age in (120,1800,3000):
            mt5_data.reset_server_clock()
            with patch.object(mt5,"symbol_info_tick",return_value=self.tick(age=age)),self.assertRaises(RuntimeError):
                mt5_data.get_tick("XAUUSD",now=self.now)
        mt5_data.reset_server_clock(self.OFFSET)
        with patch.object(mt5,"symbol_info_tick",return_value=self.tick(age=3601)),self.assertRaises(RuntimeError):
            mt5_data.get_tick("XAUUSD",now=self.now)

    def test_offset_change_needs_confirmation_and_blocks_meanwhile(self):
        mt5_data.reset_server_clock(self.OFFSET)
        with patch.object(mt5,"symbol_info_tick",return_value=self.tick(offset=2*3600)),self.assertRaises(RuntimeError):
            mt5_data.get_tick("XAUUSD",now=self.now)
        self.assertEqual(mt5_data.server_offset_seconds(),self.OFFSET)
        self.now+=mt5_data.OFFSET_CONFIRM_SECONDS
        with patch.object(mt5,"symbol_info_tick",return_value=self.tick(offset=2*3600)):
            mt5_data.get_tick("XAUUSD",now=self.now)
        self.assertEqual(mt5_data.server_offset_seconds(),2*3600)

    def test_unknown_offset_blocks_bars(self):
        with self.assertRaises(RuntimeError):
            mt5_data.rates_frame([dict(time=0,open=1,high=1,low=1,close=1)],"M15",pd.Timestamp("2026-01-01"))

    def test_env_override_pins_offset(self):
        with patch.dict("os.environ",{"MT5_SERVER_UTC_OFFSET_HOURS":"2"}):
            self.assertEqual(mt5_data.server_offset_seconds(),2*3600)

    def test_server_bars_become_utc_and_current_bar_is_excluded(self):
        mt5_data.reset_server_clock(self.OFFSET)
        now=pd.Timestamp("2026-09-23 10:16")
        rates=[dict(time=int(pd.Timestamp(t,tz="UTC").timestamp()),open=100,high=101,low=99,close=100)
               for t in ("2026-09-23 12:45","2026-09-23 13:00","2026-09-23 13:15")]  # server clock
        frame=mt5_data.rates_frame(rates,"M15",now)
        self.assertEqual(list(frame.bar_time),[pd.Timestamp("2026-09-23 09:45"),pd.Timestamp("2026-09-23 10:00")])

    def test_deal_query_and_stamp_use_server_clock(self):
        mt5_data.reset_server_clock(self.OFFSET)
        now=pd.Timestamp("2026-09-23 10:16")
        state={"account":"offline:1:USD","live":True,
            "last_bar":"2026-09-23 10:00","actions":{},"deals":{},"outcomes":{},"guards":{}}
        deal=types.SimpleNamespace(symbol=live.SYMBOL,time_msc=int(pd.Timestamp("2026-09-23 13:10",tz="UTC").timestamp()*1000),
            magic=0,comment="",order=1,ticket=1,position_id=1)
        acc=types.SimpleNamespace(server="offline",login=1,currency="USD",margin_mode=0)
        with patch.object(mt5,"account_info",return_value=acc),             patch.object(mt5,"history_deals_get",return_value=[deal]) as query,             patch.object(live,"save_state"):
            newest=live.reconcile(state,now)
        start,end=query.call_args.args
        self.assertEqual(start.replace(tzinfo=None),pd.Timestamp("2026-09-23 12:45").to_pydatetime())
        self.assertEqual(end.replace(tzinfo=None),pd.Timestamp("2026-09-23 13:17").to_pydatetime())
        self.assertEqual(newest,pd.Timestamp("2026-09-23 10:10"))

    def test_time_exit_uses_utc_position_age(self):
        mt5_data.reset_server_clock(self.OFFSET)
        opened=time.time()+self.OFFSET-(live.DEFAULTS.max_hold_minutes+1)*60  # server clock
        pos=types.SimpleNamespace(magic=live.MAGIC,sl=100,time=opened,ticket=3,volume=.1,type=0)
        info=types.SimpleNamespace(trade_tick_size=.01,point=.01,filling_mode=1,trade_exemode=2)
        state={"account":"offline:1:USD","live":True,"closes":{}}
        acc=types.SimpleNamespace(server="offline",login=1,currency="USD",margin_mode=0)
        with patch.object(mt5,"account_info",return_value=acc),patch.object(mt5,"positions_get",return_value=[pos]),             patch.object(mt5,"symbol_info_tick",return_value=self.tick()),patch.object(mt5,"symbol_info",return_value=info),             patch.object(mt5,"orders_get",return_value=()),patch.object(mt5,"order_check",return_value=types.SimpleNamespace(retcode=0)),             patch.object(mt5,"order_send",return_value=None) as send,patch.object(live,"save_state"):
            live.manage_positions(state,logging.getLogger("offline-test"))
        self.assertEqual(send.call_count,1)


class AIReviewTests(unittest.TestCase):
    """Only a genuine REJECT blocks; the service's own failures pass through (fail open)."""
    def review(self,data=None,error=None):
        from bot import ai_review
        resp=types.SimpleNamespace(raise_for_status=lambda:None,json=lambda:data)
        with patch.object(ai_review,"AI_REVIEW_ENABLED",True),             patch.object(ai_review.requests,"post",side_effect=error,return_value=resp):
            return ai_review.review_signal(symbol="XAUUSD",side="long",entry=1.,stop_loss=.9,take_profit=1.5,
                volume=.01,timeframe="M15",strategy="t",equity=1000.)

    def test_market_based_reject_blocks(self):
        self.assertFalse(self.review({"decision":"REJECT","reason":"news risk too high"}).approved)

    def test_service_data_failure_reject_passes_through(self):
        r=self.review({"decision":"REJECT","signal_id":"x","reason":"Market data unavailable, failing closed: oanda fetch failed: 401"})
        self.assertTrue(r.approved); self.assertEqual(r.decision,"UNAVAILABLE"); self.assertEqual(r.signal_id,"x")

    def test_agent_failure_reject_passes_through(self):
        self.assertTrue(self.review({"decision":"REJECT","reason":"Unified agent failed, failing closed: timeout"}).approved)

    def test_unreachable_service_passes_through(self):
        import requests
        self.assertTrue(self.review(error=requests.ConnectionError("down")).approved)


class DataRiskTests(unittest.TestCase):
    def setUp(self):
        mt5_data.reset_server_clock(0)
        self.addCleanup(mt5_data.reset_server_clock)

    def test_invalid_risk_and_minimum_volume(self):
        for value in (0,-1,float("nan"),float("inf"),.5):
            with self.assertRaises(ValueError): risk.validate_risk(value)
        info=types.SimpleNamespace(volume_min=.01,volume_max=10.,volume_step=.01)
        self.assertEqual(risk.floor_volume(.005,info),0)
        info.volume_min=info.volume_step=.001
        self.assertEqual(risk.floor_volume(.006,info),.006)

    def test_nonfinite_broker_volume_metadata_is_rejected(self):
        info=types.SimpleNamespace(volume_min=.01,volume_max=10.,volume_step=float("nan"))
        with self.assertRaises(ValueError): risk.floor_volume(1,info)

    def test_timezone_aware_snapshot_and_current_bar_exclusion(self):
        now=pd.Timestamp("2026-09-23 10:15",tz="UTC")
        rates=[dict(time=int(pd.Timestamp(t,tz="UTC").timestamp()),open=100,high=101,low=99,close=100)
               for t in ("2026-09-23 10:00","2026-09-23 10:15")]
        frame=mt5_data.rates_frame(rates,"M15",now)
        self.assertEqual(len(frame),1)

    def test_invalid_ohlc_is_rejected(self):
        rates=[dict(time=1,open=100,high=99,low=101,close=100)]
        with self.assertRaises(ValueError):
            mt5_data.rates_frame(rates,"M15",pd.Timestamp("2026-01-01"))

    def test_missing_position_response_is_not_empty(self):
        with patch.object(mt5,"positions_get",return_value=None),self.assertRaises(RuntimeError):
            mt5_data.positions()

    def test_atomic_json_leaves_old_file_on_serialization_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"state.json"
            atomic_json(path,{"a":1})
            with self.assertRaises(ValueError): atomic_json(path,{"a":float("nan")})
            self.assertEqual(json.loads(path.read_text()),{"a":1})


if __name__=="__main__":
    unittest.main()
