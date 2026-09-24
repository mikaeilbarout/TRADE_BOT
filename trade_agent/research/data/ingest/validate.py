from __future__ import annotations

from datetime import datetime, timedelta, timezone
from enum import StrEnum

import pandas as pd
from pydantic import BaseModel, Field

from research.data.ingest.base import INADMISSIBLE_PROVENANCE, Provenance

"""Validation that runs BEFORE the backtest is allowed to use a dataset.

The contract: if any check at ERROR severity fails, the dataset is not usable
and the pipeline says so. Nothing is silently repaired -- a "fix" applied to a
timestamp or a provenance label is indistinguishable from a leak once it is
written, so the answer is always to report and refuse rather than to mend.

Two checks earn special mention because they exist specifically to catch
leakage rather than sloppiness:

* **Future-dated records.** A record timestamped after the dataset was
  retrieved cannot have been available at its own timestamp. Usually a timezone
  bug; always fatal.
* **Revision leakage.** A REVISED figure whose availability timestamp is at or
  before the ORIGINAL_RELEASE it revises. A revision must, by definition, come
  later. If one does not, the vintage handling is wrong and every backtest
  using the dataset would see corrected figures.
"""


class Severity(StrEnum):
    ERROR = "ERROR"      # dataset unusable
    WARNING = "WARNING"  # usable, but the report must say this
    INFO = "INFO"


class Finding(BaseModel):
    check: str
    severity: Severity
    message: str
    affected_rows: int = 0
    examples: list[str] = Field(default_factory=list)


class ValidationReport(BaseModel):
    dataset: str
    rows: int
    findings: list[Finding] = Field(default_factory=list)
    first_timestamp: datetime | None = None
    last_timestamp: datetime | None = None

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == Severity.ERROR]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == Severity.WARNING]

    @property
    def passed(self) -> bool:
        return not self.errors

    def summary(self) -> str:
        if self.passed and not self.warnings:
            return f"{self.dataset}: {self.rows} rows, all checks passed"
        parts = [f"{self.dataset}: {self.rows} rows"]
        if self.errors:
            parts.append(f"{len(self.errors)} ERROR(S)")
        if self.warnings:
            parts.append(f"{len(self.warnings)} warning(s)")
        return ", ".join(parts)


class ValidationFailed(Exception):
    """Raised when a dataset fails validation and the caller asked to fail closed."""


def _examples(frame: pd.DataFrame, mask: pd.Series, limit: int = 3) -> list[str]:
    rows = frame[mask].head(limit)
    return [
        "; ".join(f"{col}={rows.iloc[i][col]!r}" for col in rows.columns[:4])
        for i in range(len(rows))
    ]


def validate_dataset(
    frame: pd.DataFrame,
    dataset: str,
    required_columns: tuple[str, ...],
    retrieved_at: datetime | None = None,
    expect_provenance: tuple[str, ...] | None = None,
    max_future_tolerance: timedelta = timedelta(minutes=1),
) -> ValidationReport:
    """Run every generic check over one dataset."""
    report = ValidationReport(dataset=dataset, rows=len(frame))
    if frame.empty:
        report.findings.append(
            Finding(
                check="non_empty",
                severity=Severity.WARNING,
                message=(
                    f"{dataset} contains no rows. This is reported as UNAVAILABLE "
                    "rather than treated as 'nothing happened'."
                ),
            )
        )
        return report

    # --- schema ---------------------------------------------------------
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        report.findings.append(
            Finding(
                check="required_columns",
                severity=Severity.ERROR,
                message=f"missing required column(s): {missing}",
            )
        )
        return report  # every later check assumes the schema

    # --- timestamps ------------------------------------------------------
    parsed = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    unparseable = parsed.isna()
    if unparseable.any():
        report.findings.append(
            Finding(
                check="timestamp_parseable",
                severity=Severity.ERROR,
                message=f"{int(unparseable.sum())} row(s) have unparseable timestamps",
                affected_rows=int(unparseable.sum()),
                examples=_examples(frame, unparseable),
            )
        )
        return report

    if getattr(parsed.dt, "tz", None) is None:
        report.findings.append(
            Finding(
                check="timezone_normalised",
                severity=Severity.ERROR,
                message="timestamps are not timezone-aware after UTC parsing",
            )
        )

    report.first_timestamp = parsed.min().to_pydatetime()
    report.last_timestamp = parsed.max().to_pydatetime()

    # --- chronological ordering ------------------------------------------
    if not parsed.is_monotonic_increasing:
        report.findings.append(
            Finding(
                check="chronological_order",
                severity=Severity.ERROR,
                message=(
                    "rows are not in chronological order; point-in-time queries "
                    "assume sorted input"
                ),
            )
        )

    # --- impossible timestamps -------------------------------------------
    floor = pd.Timestamp("1990-01-01", tz="UTC")
    ancient = parsed < floor
    if ancient.any():
        report.findings.append(
            Finding(
                check="impossible_timestamp",
                severity=Severity.ERROR,
                message=(
                    f"{int(ancient.sum())} row(s) are dated before {floor.date()}, "
                    "which is not a plausible publication date for this dataset"
                ),
                affected_rows=int(ancient.sum()),
                examples=_examples(frame, ancient),
            )
        )

    # --- future-dated relative to retrieval ------------------------------
    reference = retrieved_at or datetime.now(timezone.utc)
    cutoff = pd.Timestamp(reference) + pd.Timedelta(max_future_tolerance)
    future = parsed > cutoff
    if future.any():
        report.findings.append(
            Finding(
                check="future_dated",
                severity=Severity.ERROR,
                message=(
                    f"{int(future.sum())} row(s) are timestamped after the data was "
                    f"retrieved ({reference.isoformat()}). A record cannot have been "
                    "available at its own timestamp; this is normally a timezone bug "
                    "and is never repaired automatically."
                ),
                affected_rows=int(future.sum()),
                examples=_examples(frame, future),
            )
        )

    # --- retrieval ordering ----------------------------------------------
    if "retrieved_at" in frame.columns:
        retrieved = pd.to_datetime(frame["retrieved_at"], utc=True, errors="coerce")
        inverted = retrieved.notna() & (retrieved < parsed - pd.Timedelta(minutes=1))
        if inverted.any():
            report.findings.append(
                Finding(
                    check="retrieved_after_published",
                    severity=Severity.ERROR,
                    message=(
                        f"{int(inverted.sum())} row(s) claim to have been retrieved "
                        "before they were published"
                    ),
                    affected_rows=int(inverted.sum()),
                    examples=_examples(frame, inverted),
                )
            )

    # --- provenance -------------------------------------------------------
    if "provenance" in frame.columns:
        provenance = frame["provenance"].fillna("").astype(str).str.upper()
        unknown = provenance.isin(("", "UNKNOWN"))
        if unknown.any():
            report.findings.append(
                Finding(
                    check="provenance_present",
                    severity=Severity.WARNING,
                    message=(
                        f"{int(unknown.sum())} row(s) have no provenance. They cannot "
                        "be treated as point-in-time safe."
                    ),
                    affected_rows=int(unknown.sum()),
                )
            )
        if expect_provenance:
            allowed = {value.upper() for value in expect_provenance}
            unexpected = ~provenance.isin(allowed) & ~unknown
            if unexpected.any():
                report.findings.append(
                    Finding(
                        check="provenance_allowed",
                        severity=Severity.ERROR,
                        message=(
                            f"{int(unexpected.sum())} row(s) carry provenance outside "
                            f"{sorted(allowed)} for this dataset: "
                            f"{sorted(set(provenance[unexpected]))}"
                        ),
                        affected_rows=int(unexpected.sum()),
                    )
                )
    else:
        report.findings.append(
            Finding(
                check="provenance_present",
                severity=Severity.ERROR,
                message="dataset has no provenance column, so it cannot be trusted",
            )
        )

    # --- source metadata --------------------------------------------------
    for column in ("source", "source_id"):
        if column in frame.columns:
            blank = frame[column].isna() | (frame[column].astype(str).str.strip() == "")
            if blank.any():
                report.findings.append(
                    Finding(
                        check=f"{column}_present",
                        severity=Severity.WARNING,
                        message=f"{int(blank.sum())} row(s) have no {column}",
                        affected_rows=int(blank.sum()),
                    )
                )

    # --- duplicates -------------------------------------------------------
    if "source_id" in frame.columns and "source" in frame.columns:
        duplicated = frame.duplicated(subset=["source", "source_id"], keep=False)
        if duplicated.any():
            report.findings.append(
                Finding(
                    check="duplicate_records",
                    severity=Severity.ERROR,
                    message=(
                        f"{int(duplicated.sum())} row(s) share a (source, source_id); "
                        "deduplication did not run or the ids are not unique"
                    ),
                    affected_rows=int(duplicated.sum()),
                )
            )

    return report


def validate_calendar_vintages(frame: pd.DataFrame) -> list[Finding]:
    """Checks that only make sense for a vintage-aware calendar.

    The one that matters: a revision must become available strictly after the
    original release it revises. A revision that appears at or before its
    original means the vintage handling is wrong, and every backtest reading the
    dataset would be seeing corrected figures at the original release time.
    """
    findings: list[Finding] = []
    if frame.empty or "provenance" not in frame.columns:
        return findings
    if not {"series_id", "reference_period"}.issubset(frame.columns):
        return findings

    work = frame.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
    work["provenance"] = work["provenance"].astype(str).str.upper()

    originals = work[work["provenance"] == Provenance.ORIGINAL_RELEASE.value]
    revisions = work[work["provenance"] == Provenance.REVISED.value]
    if originals.empty or revisions.empty:
        return findings

    first_release = originals.groupby(["series_id", "reference_period"])["timestamp"].min()
    leaks: list[str] = []
    for _, row in revisions.iterrows():
        key = (row["series_id"], row["reference_period"])
        original_at = first_release.get(key)
        if original_at is None or pd.isna(original_at):
            continue
        if row["timestamp"] <= original_at:
            leaks.append(
                f"{row['series_id']} {row['reference_period']}: revision at "
                f"{row['timestamp']} is not after original release at {original_at}"
            )

    if leaks:
        findings.append(
            Finding(
                check="revision_after_original",
                severity=Severity.ERROR,
                message=(
                    f"{len(leaks)} revision(s) are not strictly later than the "
                    "original release they revise. A backtest reading this dataset "
                    "would see revised figures at original-release time."
                ),
                affected_rows=len(leaks),
                examples=leaks[:3],
            )
        )

    # An original release should never be missing for a period that has revisions.
    revision_keys = set(
        zip(revisions["series_id"], revisions["reference_period"], strict=False)
    )
    original_keys = set(first_release.index)
    orphans = revision_keys - original_keys
    if orphans:
        findings.append(
            Finding(
                check="revision_without_original",
                severity=Severity.WARNING,
                message=(
                    f"{len(orphans)} reference period(s) have revisions but no "
                    "original release in the dataset. The original may predate the "
                    "requested window; the revised value must not be used as the "
                    "original."
                ),
                affected_rows=len(orphans),
                examples=[f"{s} {p}" for s, p in sorted(orphans)[:3]],
            )
        )
    return findings


def validate_sentiment_provenance(
    frame: pd.DataFrame, point_in_time_only: bool = True
) -> list[Finding]:
    """Checks specific to sentiment, where provenance decides admissibility.

    When `point_in_time_only` (the default for the leakage-safe experiment), any
    RETROSPECTIVE row is an ERROR: such a value was produced today from an old
    document, and the whole design rests on not treating it as evidence that was
    available at the time.
    """
    findings: list[Finding] = []
    if frame.empty or "provenance" not in frame.columns:
        return findings

    provenance = frame["provenance"].fillna("").astype(str).str.upper()
    inadmissible = provenance.isin(INADMISSIBLE_PROVENANCE)
    if inadmissible.any() and point_in_time_only:
        findings.append(
            Finding(
                check="sentiment_point_in_time_only",
                severity=Severity.ERROR,
                message=(
                    f"{int(inadmissible.sum())} sentiment row(s) carry provenance "
                    f"{sorted(set(provenance[inadmissible]))}, which is not "
                    "point-in-time. Retrospective sentiment must live in its own "
                    "dataset and must not be loaded for the leakage-safe experiment."
                ),
                affected_rows=int(inadmissible.sum()),
            )
        )

    # A model-generated value must say which model produced it, or the record
    # cannot be reproduced or audited later.
    if "method" in frame.columns and "model" in frame.columns:
        generated = provenance == Provenance.RETROSPECTIVE.value
        missing_model = generated & (
            frame["model"].isna() | (frame["model"].astype(str).str.strip() == "")
        )
        if missing_model.any():
            findings.append(
                Finding(
                    check="generated_sentiment_names_its_model",
                    severity=Severity.WARNING,
                    message=(
                        f"{int(missing_model.sum())} retrospective row(s) do not name "
                        "the model that produced them"
                    ),
                    affected_rows=int(missing_model.sum()),
                )
            )
    return findings


NEWS_REQUIRED = ("timestamp", "source", "headline", "category", "provenance")
CALENDAR_REQUIRED = ("timestamp", "name", "importance", "provenance")
SENTIMENT_REQUIRED = ("timestamp", "source", "value", "provenance")


def validate_news(frame: pd.DataFrame, retrieved_at: datetime | None = None) -> ValidationReport:
    return validate_dataset(
        frame,
        "news",
        NEWS_REQUIRED,
        retrieved_at=retrieved_at,
        expect_provenance=(Provenance.ORIGINAL_RELEASE.value,),
    )


def validate_calendar(
    frame: pd.DataFrame, retrieved_at: datetime | None = None
) -> ValidationReport:
    report = validate_dataset(
        frame,
        "economic_calendar",
        CALENDAR_REQUIRED,
        retrieved_at=retrieved_at,
        expect_provenance=(
            Provenance.ORIGINAL_RELEASE.value,
            Provenance.REVISED.value,
        ),
    )
    report.findings.extend(validate_calendar_vintages(frame))
    return report


def validate_sentiment(
    frame: pd.DataFrame,
    retrieved_at: datetime | None = None,
    point_in_time_only: bool = True,
) -> ValidationReport:
    expected = (
        (Provenance.POINT_IN_TIME_CAPTURE.value,)
        if point_in_time_only
        else (Provenance.POINT_IN_TIME_CAPTURE.value, Provenance.RETROSPECTIVE.value)
    )
    report = validate_dataset(
        frame,
        "sentiment",
        SENTIMENT_REQUIRED,
        retrieved_at=retrieved_at,
        expect_provenance=expected,
    )
    report.findings.extend(
        validate_sentiment_provenance(frame, point_in_time_only=point_in_time_only)
    )
    return report


def fail_closed(reports: list[ValidationReport]) -> None:
    """Raise unless every dataset passed.

    Called before the backtest is allowed to run. The message names every
    failing check, because "validation failed" on its own is not actionable.
    """
    failing = [report for report in reports if not report.passed]
    if not failing:
        return
    lines: list[str] = []
    for report in failing:
        for finding in report.errors:
            lines.append(f"[{report.dataset}] {finding.check}: {finding.message}")
            for example in finding.examples:
                lines.append(f"    e.g. {example}")
    raise ValidationFailed(
        "historical data validation failed; the backtest is blocked rather than "
        "run on data that may leak:\n  - " + "\n  - ".join(lines)
    )
