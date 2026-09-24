from __future__ import annotations

from datetime import datetime

from app.agents.base import AgentRun, BaseAgent
from app.agents.context import full_technical_conditions, signal_payload
from app.models.agent_decision import TechnicalAgentResult
from app.models.market_data import MarketSnapshot
from app.models.signal import TradeSignal


class TechnicalAgent(BaseAgent):
    """One of three specialist agents (news, sentiment, technical) that run
    concurrently -- signal + full multi-timeframe market data, independent
    of the other two. Only the final decision agent sees all three
    findings together.

    Parallelized 2026-09-18, see sentiment_agent.py's docstring for why.

    It receives the OHLCV tail per timeframe (not just indicator values) so
    it derives structure itself rather than trusting anyone's summary --
    including the originating bot's.
    """

    prompt_file = "technical_agent.md"
    response_model = TechnicalAgentResult
    agent_name = "technical_agent"
    task_instruction = (
        "Independently verify this technical setup from the market data provided. "
        "Do not assume the bot's signal is sound. Propose a modification if the "
        "directional idea is valid but entry/SL/TP is not."
    )

    def build_payload(
        self,
        signal: TradeSignal,
        market: MarketSnapshot,
        now: datetime | None = None,
    ) -> dict:
        return {
            "chain_position": "1 of 3 parallel specialists (technical) -> final decision",
            "signal": signal_payload(signal, now),
            "market_conditions": full_technical_conditions(market),
        }

    async def run(
        self,
        signal: TradeSignal,
        market: MarketSnapshot,
        now: datetime | None = None,
    ) -> AgentRun:
        return await self._call(self.build_payload(signal, market, now))
