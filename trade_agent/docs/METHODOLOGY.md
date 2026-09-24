# Experiment methodology

This document states what the experiment does, what it deliberately refuses to
do, and what its results cannot support. It is written to be read BEFORE the
numbers, because several of the limitations below change how the numbers should
be interpreted.

## The question

Does inserting a four-agent AI review layer between a mechanical strategy's
signals and their execution improve risk-adjusted results, net of the AI's own
cost?

## Design

A chronological 70/30 split over XAUUSD 15-minute bars:

| Period | Bars | Use |
|---|---|---|
| First 70% | development | all analysis, all parameter selection |
| Final 30% | out-of-sample | tested ONCE, with frozen parameters |

The split boundary is computed from the data's own bar count, never
hard-coded, and 200 bars of embargo are dropped at the boundary so indicator
state warmed up on development data cannot bleed into the first test trades.

### Why the split is enforced in code, not by discipline

`research/data/split.py` refuses out-of-sample access unless a `StrategySeal`
exists on disk. The seal records the chosen parameters, their hash, the
development window and metrics, the optimizer configuration, the candidate
count, the selection criterion and the dataset hash. Reading out-of-sample
data verifies the parameters against the sealed hash, so testing anything
other than the frozen strategy raises rather than proceeding.

Re-optimizing after a seal exists requires passing an explicit flag, and doing
so is appended to the seal's `reseal_history` — so "we tuned it after seeing
the test set" cannot happen silently.

### Parameter selection

`research/strategy/optimizer.py` runs expanding-window walk-forward folds
INSIDE the development set. It takes the `DataSplit` object, not a bar frame,
so there is no argument through which out-of-sample bars could arrive.

Selection is NOT by highest total profit — with a few hundred candidates, some
parameter set always wins one window by luck. Eligibility filters are applied
first (minimum trades per fold, minimum trades overall, a minimum fraction of
profitable folds, a drawdown cap), and the winner is chosen by a robustness
objective: the MEDIAN per-fold return-per-unit-of-drawdown, minus a penalty on
the spread between folds. Two candidates with the same median are separated by
consistency. If no candidate clears the filters, nothing is selected and no
seal is written.

## The two arms

| | Experiment A | Experiment B |
|---|---|---|
| Strategy signals | same | same |
| Deterministic pre-trade gate | yes | yes |
| AI decision layer | no | yes |
| Execution guard | yes | yes |
| Fill model, costs, sizing, limits | same | same |

Both arms run through ONE executor (`research/backtest/executor.py`), called
with and without decisions. Two separately written runners would drift, and a
comparison between two drifted runners measures the drift.

`research/experiment.py` builds the engine, risk service, executor and policy
thresholds for both arms from one `ExperimentConfig`, and
`assert_ab_identical` proves capital, risk, spread, slippage, commission,
position limits, daily limits, instrument settings, execution assumptions and
strategy parameters are the same in both. `assert_manifests_match` then
compares the two run manifests and fails on any difference outside the
explicitly AI-only fields. Datasets are allowed to differ (the AI arm consumes
news and sentiment the baseline never touches), but any dataset named in BOTH
manifests — the candle series above all — must be byte-identical.

## Execution assumptions

Recorded in every manifest with a semantics version, so a change to the fill
model cannot silently invalidate an old result:

- A signal on the close of bar *i* is filled at the OPEN of bar *i+1*. A
  strategy never trades at the price that triggered it.
- Fills cross the spread, using the spread recorded in the tick data.
- When a bar contains both the stop and the target, the STOP is taken. Without
  tick replay the order is unknown, and assuming the favourable one is the most
  common way a backtest flatters itself.
- Stops fill worse than their trigger; targets fill at the level.
- A bar that gaps through a level fills at the open, not the level.
- An AI-modified entry away from the signal price is a RESTING LIMIT order: it
  fills at the limit price only when the bar's spread-adjusted extreme reaches
  it, it is cancelled if the take-profit is reached first, and it expires
  unfilled after a configured number of bars. An unfilled order is recorded as
  a skipped signal, never as a fill at a later price.
- Exit resolution starts on the fill bar itself, so a bar that touches the
  entry and the stop is a loss.

## Counterfactual scoring

Every signal the AI declines is simulated on the same engine, with the same
fill assumptions, at the ORIGINAL levels — so "how many profitable trades did
the AI reject" is answerable. These are sized off a fixed notional balance and
never added to either arm's equity curve: they are diagnostic and cannot move
a reported return by a cent.

A rejected trade that would have won is not proof the rejection was wrong.
Declining a trade that eventually won after running 2R against the entry can be
correct risk management, which is why maximum adverse excursion is reported
alongside every counterfactual outcome.

## Automated historical data ingestion

News, economic-calendar and sentiment data are fetched automatically by
`python -m research_data`; none of it is hand-prepared. Two sources are
implemented, chosen from a survey recorded in
`research/data/ingest/registry.py` (which also records what was rejected and
why):

| | Source | Cost | Key | Coverage | Point-in-time basis |
|---|---|---|---|---|---|
| News, sentiment | GDELT 2.0 GKG 15-minute archive files | Free | none | 2015-02-18 onward | The file NAME is its publication slot; the provider publishes an MD5 per file |
| Calendar | FRED / ALFRED vintages | Free | `FRED_API_KEY` | series-dependent | `output_type=4` gives the initial release, `output_type=3` the revisions |

### Why archive files rather than a search API

GDELT also offers a full-text search API. It is not used, because it answers
"what does the index say today", which is the reconstruction this experiment
exists to avoid. The 15-minute archive files are immutable and their filenames
are timestamps, so "what was knowable at 14:45" is a file, not an inference.

### Availability is the later of publication and discovery

Every ingested record carries `published_at` (what the source says) and
`discovered_at` (when the aggregator exposed it), and point-in-time filtering
uses `available_at = max(published_at, discovered_at)`. An article published at
14:32 that appeared in GDELT's 14:45 file could not have been read through GDELT
at 14:33, so treating 14:32 as its availability would hand the backtest up to
fifteen minutes of lookahead. The maximum is never earlier than publication, so
it also satisfies the stricter rule `publication_time <= T`.

### Original releases versus revisions

FRED's archival side distinguishes the first published value for a period from
every later correction, and both are fetched into SEPARATE records:
`ORIGINAL_RELEASE` and `REVISED`. The point-in-time layer drops every `REVISED`
row at load time, so no code path can return a corrected figure as the value
that was public at the original release. Validation additionally refuses a
dataset in which any revision is not strictly later than the release it revises.

### The release-time limitation, stated plainly

**ALFRED records the release DATE, not the clock time.** It knows CPI for May
became available on 2023-06-13; it does not know 08:30. A 15-minute backtest
needs a time, so there are two policies and both are labelled:

* `END_OF_DAY` (default) — available at 23:59:59 UTC on the release date.
  Never reveals a figure before it was published; hides it for the rest of the
  release day. `time_precision=DATE_ONLY`.
* `SCHEDULED_LOCAL` — the publisher's long-standing clock time (08:30
  America/New_York for BLS releases), converted to UTC with DST handled.
  Realistic but an assumption, so `time_precision=IMPUTED_FROM_SCHEDULE`.

Neither invents a value; they differ only in when an existing value becomes
visible. The default is the conservative one.

### Sentiment: three states, never blurred

* `POINT_IN_TIME_CAPTURE` — GDELT's own tone scores, which GDELT computed at
  ingestion and published inside the 15-minute file. The claim is narrow and
  checkable: the NUMBER existed publicly at that timestamp. This is the only
  sentiment the leakage-safe experiment uses.
* `RETROSPECTIVE` — archived headlines scored by a model running today. Honest,
  useful as a diagnostic comparison, and inadmissible: the model reading a 2023
  headline knows what happened next. Written to a separate file, refused by the
  loader, and off by default.
* `UNAVAILABLE` — no data. The dependent agent answers UNAVAILABLE and the
  fail-closed policy applies.

The distinction is enforced by construction rather than by discipline: each
builder can emit only one provenance value, hard-wired as a class attribute, and
the point-in-time builder raises if handed a retrospective record.

### Ingestion is resumable and cached

Progress is committed per window (a 15-minute slot for GDELT, a series-year for
FRED). A window is `done`, `empty`, or `failed`; the first two are never
re-fetched and `failed` is retried next run — so a transient error cannot become
a permanent hole that looks like a quiet period. Downloaded bytes are cached at
a deterministic path with a SHA-256, verified against the provider's own MD5
where one is published. A five-year backfill that dies halfway resumes.

### Cost of ingestion

The news corpus needs no LLM at all: relevance filtering is deterministic
keyword and GDELT-theme matching, applied before anything else, which is why a
firehose of ~100k articles a day reduces to a usable corpus for free. The only
component that can spend money is the optional retrospective sentiment builder,
and it batches one call per time slot (not per article), caches by content hash,
uses the cheap model with a three-field output, and will not run without an
explicit flag.

## Data policy

Missing data is reported as missing. Nothing is generated, back-filled, or
substituted:

- **News** requires timestamp, source, headline and category, and is served
  only for timestamps at or after publication.
- **Sentiment** must evidence point-in-time capture. Scoring archived text with
  a present-day model is refused (`RETROSPECTIVE_SCORING`): it produces today's
  reading of old text, which is hindsight wearing a timestamp.
- **Economic calendar** serves the SCHEDULE (known in advance) but withholds a
  released value until its release time, and excludes REVISED figures entirely
  — the market at the time traded the original print.

An agent whose dataset has no coverage for a timestamp answers UNAVAILABLE and
the fail-closed policy applies. The agent is not called at all, because paying
a model to say "I have no data" is waste.

## Cost control

- Deterministic rejection runs first and costs nothing.
- An agent whose point-in-time store is empty is skipped.
- The three analysts run on the cheapest model; only the adjudicator uses a
  more capable one. Every model is an environment variable.
- Static prompt prefixes are cached; all four token counts are recorded
  separately, because the point of caching is to see the split.
- Outputs are closed-vocabulary reason codes, not prose.
- A hard USD budget stops the run BEFORE a call that would exceed it, and
  spend is recovered from the checkpoint so a restart cannot double it.

The pilot's cost report is measured, never estimated. Projections are explicit
multiples of the measured average and labelled as such.


## Known limitations

### CRITICAL - The language model may know what happened after the signal timestamp

Every agent in this experiment is a language model trained on text published after the period being replayed. When it is shown a XAUUSD setup dated 2024-03-08, it may recognise the period and carry knowledge of what gold did next: rate decisions, geopolitical events, the shape of the trend. That knowledge is not in the prompt and cannot be removed from the weights. Any measured improvement from the AI layer is therefore an UPPER BOUND on what the same layer would achieve on genuinely unseen future data.

**Why this is not mitigated:** The prompts instruct the agents to reason only from the supplied data, and the point-in-time stores make sure no future data is supplied. Neither addresses this. An instruction cannot remove information from a model's weights, and a model does not need a future headline in its context to recall the period. Claiming that prompt discipline eliminates this risk would be false; it reduces explicit look-ahead, which is a different problem. Nor is the risk removed by the fact that the model is not asked to predict prices: recognising 'this is the week before the March 2024 breakout' is enough to bias a verdict.

**What would resolve it:** Three controls, in increasing strength: (1) a DATE-STRIPPED PLACEBO ARM -- re-run the pilot with all absolute dates and period-identifying details removed from agent payloads (`AI_STRIP_DATES=true`) and compare decision distributions; a large difference indicates the dates were carrying information. (2) A SHUFFLED-PERIOD CONTROL -- present real setups with dates from unrelated periods and check whether verdict quality tracks the true period. (3) The only conclusive test: FORWARD PAPER TRADING on data generated after the model's training cutoff. Until (3) runs, the out-of-sample result is evidence about the layer's ceiling, not its expected live performance.

_Affects: experiment_b, ai_vs_baseline_comparison, all_ai_decisions_

### CRITICAL - Agents whose point-in-time dataset is missing cannot contribute

An agent with no data for a timestamp answers UNAVAILABLE and the fail-closed policy applies. If news and sentiment datasets are absent, the four-agent chain is effectively a technical agent plus an adjudicator, and the result measures THAT system -- not the one the design describes.

**Why this is not mitigated:** Deliberately. The alternative -- generating plausible historical headlines or scoring archived text with a present-day model -- would produce numbers that look complete and mean nothing.

**What would resolve it:** Supply genuine point-in-time datasets: timestamped news with publication times, sentiment captured live (not retrospectively scored), and an economic calendar with original-release values.

_Affects: experiment_b, news_agent, sentiment_agent_

### MATERIAL - Fills are resolved on 15-minute bars, not ticks

When one bar's range contains both the stop and the target, the true order of events is unknown without tick replay. The engine always assumes the stop came first, and assumes the pessimistic side of every other ambiguity.

**Why this is not mitigated:** Tick-level replay is implementable -- the tick data is the source of the bars -- but it was deliberately deferred until the pilot infrastructure is proven. The current assumption is conservative, so it understates rather than flatters results.

**What would resolve it:** Replay the stored ticks inside any bar where both levels are touched, and compare the two sets of results to size the effect.

_Affects: experiment_a, experiment_b, counterfactuals_

### MATERIAL - One instrument, one strategy, one out-of-sample period

The result describes this strategy on XAUUSD over one contiguous test period. It is a single observation, not a distribution, and the test period has its own regime.

**Why this is not mitigated:** It is inherent to the 70/30 design, which is the correct design for the question asked: it buys one honest out-of-sample answer at the cost of statistical breadth.

**What would resolve it:** Repeat on other instruments and other strategies, and report the distribution of outcomes rather than one number.

_Affects: all_conclusions_

### MATERIAL - A 100-signal pilot measures cost reliably and performance barely

The pilot exists to measure cost per signal and to prove the pipeline. With roughly 100 signals, a difference in win rate or expectancy between the arms is well inside noise.

**Why this is not mitigated:** It is the point of a pilot: measure the bill and validate the machinery before spending on the full run.

**What would resolve it:** The full out-of-sample run, once the pilot's measured cost justifies it.

_Affects: pilot_results_

### MINOR - Overnight financing is not charged

Commission, spread and slippage are modelled; swap/financing on positions held overnight is not. The strategy's holding periods are intraday-to-short, so the omission is small, but it is an omission and it flatters both arms equally.

**Why this is not mitigated:** Deliberately deferred until after the pilot.

**What would resolve it:** Add a per-night financing charge from the broker's published rates, applied to both arms.

_Affects: experiment_a, experiment_b_

## What this experiment cannot tell you

- Whether the AI layer will work on future data. See
  `MODEL_KNOWLEDGE_LEAKAGE` above; only forward paper trading past the model's
  training cutoff can answer that.
- Whether the strategy is good. The strategy is a vehicle for measuring the
  AI layer, and one instrument over one test period is a single observation.
- Whether the full four-agent design works, if news and sentiment datasets are
  absent. In that case the result describes a technical agent plus an
  adjudicator, and the report says so.

## Reproducing a run

```
# --- historical data ingestion (automated) ---
python -m research_data sources                    # what is needed, and why
python -m research_data fetch-news                 # GDELT archives (resumable)
python -m research_data fetch-calendar             # FRED vintages (needs a free key)
python -m research_data build-sentiment            # point-in-time tone, no model
python -m research_data validate                   # fail closed on any problem
python -m research_data status                     # coverage and gaps

# --- the experiment ---
python -m research.cli config                     # write the experiment config
python -m research.cli fetch                      # or: fetch --csv <your export>
python -m research.cli candles                     # ticks -> M15 bars
python -m research.cli datasets                    # audit PIT data availability
python -m research.cli split                       # report the boundary (still locked)
python -m research.cli develop                     # search the first 70% ONLY
python -m research.cli freeze                      # seal; unlocks out-of-sample
python -m research.cli baseline                    # Experiment A
python -m research.cli pilot --count 100 --mock    # rehearse with no spend
python -m research.cli pilot --count 100           # Experiment B, measured cost
python -m research.cli compare                     # A vs B + counterfactuals
```

Every command reads the same experiment config and writes where the next one
expects. Ordering is enforced by the artifacts: `baseline` cannot read
out-of-sample data until `freeze` has written a seal, and `freeze` refuses to
run without a `develop` report.
