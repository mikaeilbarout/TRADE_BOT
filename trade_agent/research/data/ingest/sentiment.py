from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from research.data.ingest.base import (
    NewsRecord,
    Provenance,
    SentimentRecord,
    TimePrecision,
    now_utc,
    stable_id,
)

"""Two sentiment builders, and the line between them.

This is the part of the pipeline where it is easiest to produce something that
looks like point-in-time data and is not. So the distinction is enforced by
construction: each builder can only emit one provenance value, and the value is
hard-wired, not a parameter the caller chooses.

**`PointInTimeToneBuilder` -- POINT_IN_TIME_CAPTURE.**
Aggregates the tone scores GDELT itself computed and published inside a
15-minute archive file stamped with that slot. The claim is narrow and checkable:
the NUMBER existed, publicly, at that timestamp. Nothing is recomputed.

**`RetrospectiveLlmBuilder` -- RETROSPECTIVE.**
Scores archived headlines with a model running today. Useful as a diagnostic
comparison, and inadmissible as point-in-time evidence, because the model
reading a 2023 headline in 2026 knows what happened next. It is labelled
RETROSPECTIVE, the existing dataset loader refuses it for the leakage-safe
experiment, and it is off by default.

The two are never merged into one file. A single sentiment dataset mixing them
would be indistinguishable from the honest one after the fact.
"""


@dataclass
class ToneAggregation:
    """How published tone values are collapsed into one observation.

    Aggregating within the publication slot only -- never across a window that
    extends past it -- is what keeps the result point-in-time. A rolling mean
    that reached forward would be a different, leaking statistic.
    """

    slot_minutes: int = 15
    min_articles: int = 1
    # Tone is roughly [-100, 100]; dividing by 10 puts ordinary news in [-1, 1]
    # and clamps the tails rather than letting one furious article dominate.
    normalise_divisor: float = 10.0


class PointInTimeToneBuilder:
    """Sentiment from values that were published at their timestamp."""

    provenance = Provenance.POINT_IN_TIME_CAPTURE
    method = "gdelt_v15_tone_slot_mean"

    def __init__(self, aggregation: ToneAggregation | None = None) -> None:
        self._agg = aggregation or ToneAggregation()

    def from_tone_records(self, records: list[SentimentRecord]) -> list[SentimentRecord]:
        """Pass through GDELT-derived tone records, asserting their provenance.

        The records arrive already built by the GDELT source (which has the raw
        tone column); this builder exists to enforce the invariant in one place
        and to reject anything that reaches it with the wrong provenance.
        """
        for record in records:
            if record.provenance != Provenance.POINT_IN_TIME_CAPTURE:
                raise ValueError(
                    f"{record.source_id}: PointInTimeToneBuilder received a record "
                    f"with provenance {record.provenance.value}. Only values that "
                    "were themselves published at their timestamp may carry "
                    "POINT_IN_TIME_CAPTURE."
                )
        return records


# --- retrospective -------------------------------------------------------
SENTIMENT_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {
            "type": "integer",
            "minimum": -100,
            "maximum": 100,
            "description": (
                "Sentiment for GOLD (XAUUSD) implied by these headlines: -100 "
                "strongly bearish for gold, 0 neutral, +100 strongly bullish."
            ),
        },
        "confidence": {
            "type": "integer",
            "minimum": 0,
            "maximum": 100,
            "description": "How clear the signal is across the headlines.",
        },
        "driver": {
            "type": "string",
            "enum": [
                "RATES", "INFLATION", "USD", "RISK_SENTIMENT", "GEOPOLITICS",
                "SUPPLY_DEMAND", "MIXED", "NONE",
            ],
        },
    },
    "required": ["score", "confidence", "driver"],
    "additionalProperties": False,
}

RETROSPECTIVE_PROMPT = """# Retrospective headline sentiment

You are scoring archived news headlines for their implication for the gold price
(XAUUSD). Output a single score per batch of headlines.

Scoring:
- -100 to -34: headlines that argue for a LOWER gold price (rising real rates, a
  stronger dollar, hawkish policy, risk-on).
- -33 to +33: neutral, mixed, or not gold-relevant.
- +34 to +100: headlines that argue for a HIGHER gold price (falling real rates,
  a weaker dollar, dovish policy, risk-off, geopolitical stress).

Rules:
- Score ONLY what the headlines say. Do not use any knowledge of what happened
  afterwards, and do not reason about the date.
- Headlines are often ambiguous. A middling score with low confidence is the
  correct answer far more often than a strong one.
- Return the structured output only.

These scores are recorded as RETROSPECTIVE and are excluded from the
leakage-safe experiment. They exist as a diagnostic comparison against
point-in-time tone, not as a substitute for it."""


@dataclass
class RetrospectiveBatch:
    """One unit of LLM work: the headlines inside one aggregation slot."""

    slot_start: datetime
    slot_end: datetime
    headlines: list[str]
    source_ids: list[str]
    custom_id: str

    @property
    def prompt_payload(self) -> dict:
        return {"headlines": self.headlines[:20]}


@dataclass
class RetrospectiveLlmBuilder:
    """Scores archived headlines with a model running now.

    Cost discipline, because this is the one place in the ingestion pipeline
    that can spend money:

    * Headlines are deduplicated and relevance-filtered upstream, so the model
      never sees the raw firehose.
    * Batching -- one call per time slot, not per article. A slot with 20
      headlines is one request.
    * Results are cached on disk by content hash, so re-running never re-scores
      a slot whose headlines have not changed.
    * The cheap model, and a tiny structured output (three fields).
    * Nothing runs unless explicitly enabled.
    """

    model: str = "claude-haiku-4-5"
    cache_path: Path | None = None
    aggregation: ToneAggregation = field(default_factory=ToneAggregation)
    # Holds scores when no cache_path is configured, so the builder behaves the
    # same way with and without a file on disk. Without this, scores written by
    # a path-less builder vanished silently.
    _memory: dict = field(default_factory=dict, repr=False)
    provenance = Provenance.RETROSPECTIVE
    method = "llm_retrospective_headline_scoring"

    def batches(self, news: list[NewsRecord]) -> list[RetrospectiveBatch]:
        """Group headlines into slot-sized batches.

        Slots are aligned to the aggregation grid so a batch maps to exactly one
        sentiment observation, and the batch id is a content hash so an
        unchanged batch hits the cache.
        """
        buckets: dict[datetime, list[NewsRecord]] = {}
        step = timedelta(minutes=self.aggregation.slot_minutes)
        for record in sorted(news, key=lambda r: r.available_at):
            moment = record.available_at
            floor = moment.replace(
                minute=(moment.minute // self.aggregation.slot_minutes)
                * self.aggregation.slot_minutes,
                second=0,
                microsecond=0,
            )
            buckets.setdefault(floor, []).append(record)

        out: list[RetrospectiveBatch] = []
        for slot, records in sorted(buckets.items()):
            if len(records) < self.aggregation.min_articles:
                continue
            headlines = [r.headline for r in records]
            out.append(
                RetrospectiveBatch(
                    slot_start=slot,
                    slot_end=slot + step,
                    headlines=headlines,
                    source_ids=[r.source_id for r in records],
                    # Content-addressed: identical headlines produce the same id,
                    # so a re-run is a cache hit rather than a new charge.
                    custom_id=stable_id("retro", slot.isoformat(), *sorted(headlines)),
                )
            )
        return out

    # --- cache ------------------------------------------------------------
    def _load_cache(self) -> dict:
        if self.cache_path is None:
            return dict(self._memory)
        if not Path(self.cache_path).exists():
            return {}
        return json.loads(Path(self.cache_path).read_text())

    def _save_cache(self, payload: dict) -> None:
        if self.cache_path is None:
            self._memory = dict(payload)
            return
        Path(self.cache_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.cache_path).write_text(json.dumps(payload, indent=2, default=str))

    def cached_scores(self) -> dict:
        return self._load_cache()

    def pending(self, batches: list[RetrospectiveBatch]) -> list[RetrospectiveBatch]:
        """Batches not already scored, so a re-run costs nothing for old slots."""
        cached = self._load_cache()
        return [batch for batch in batches if batch.custom_id not in cached]

    def store_scores(self, scores: dict[str, dict]) -> None:
        cached = self._load_cache()
        cached.update(scores)
        self._save_cache(cached)

    def to_records(
        self, batches: list[RetrospectiveBatch], retrieved_at: datetime | None = None
    ) -> list[SentimentRecord]:
        """Turn cached scores into RETROSPECTIVE sentiment records.

        The provenance is the class attribute, not an argument: a caller cannot
        promote these to POINT_IN_TIME_CAPTURE by passing a flag.
        """
        cached = self._load_cache()
        retrieved_at = retrieved_at or now_utc()
        records: list[SentimentRecord] = []
        for batch in batches:
            payload = cached.get(batch.custom_id)
            if not payload:
                continue
            score = float(payload["score"])
            records.append(
                SentimentRecord(
                    source="llm_retrospective",
                    source_id=batch.custom_id,
                    value=max(-1.0, min(1.0, score / 100.0)),
                    raw_value=score,
                    scale="llm_score_-100_100",
                    method=self.method,
                    model=payload.get("model", self.model),
                    article_count=len(batch.headlines),
                    confidence=(
                        float(payload["confidence"]) / 100.0
                        if payload.get("confidence") is not None
                        else None
                    ),
                    published_at=batch.slot_start,
                    discovered_at=batch.slot_start,
                    retrieved_at=retrieved_at,
                    time_precision=TimePrecision.EXACT,
                    # Hard-wired. This value was produced today from an old
                    # document and must never be treated as point-in-time.
                    provenance=Provenance.RETROSPECTIVE,
                    original_release=False,
                )
            )
        return records

    def cost_estimate(self, batches: list[RetrospectiveBatch]) -> dict:
        """Rough cost before spending anything.

        Deliberately an estimate from token arithmetic, clearly labelled -- the
        measured figure comes from the run's own usage reporting.
        """
        pending = self.pending(batches)
        prompt_tokens = len(RETROSPECTIVE_PROMPT) // 4
        per_call_input = prompt_tokens + 200
        per_call_output = 40
        # claude-haiku-4-5: $1.00 / MTok input, $5.00 / MTok output.
        cost = len(pending) * (
            per_call_input * 1.00 / 1_000_000 + per_call_output * 5.00 / 1_000_000
        )
        return {
            "batches_total": len(batches),
            "batches_pending": len(pending),
            "batches_cached": len(batches) - len(pending),
            "model": self.model,
            "estimated_usd": round(cost, 4),
            "estimated_usd_with_batch_api_discount": round(cost * 0.5, 4),
            "basis": (
                "estimated from prompt length and a 40-token structured output; "
                "the measured cost comes from the run's usage reporting"
            ),
        }


def sentiment_frame(records: list[SentimentRecord]) -> pd.DataFrame:
    """Rows for the point-in-time sentiment store."""
    return pd.DataFrame([record.to_row() for record in records])


def split_by_provenance(
    records: list[SentimentRecord],
) -> dict[str, list[SentimentRecord]]:
    """Separate point-in-time from retrospective records.

    Used when writing: the two go to different files, because one dataset
    holding both would make the distinction unrecoverable.
    """
    grouped: dict[str, list[SentimentRecord]] = {}
    for record in records:
        grouped.setdefault(record.provenance.value, []).append(record)
    return grouped
