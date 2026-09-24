# Multi-Agent AI Trading Decision System — XAUUSD

A four-agent AI review layer that sits between a mechanical trading bot's signals
and their execution, plus the out-of-sample research stack that measures whether
it actually helps.

Two things live here:

| | What it is | Entry point |
|---|---|---|
| **`app/`** | A FastAPI service that reviews live signals through a News → Sentiment → Technical → Final Decision agent chain and returns APPROVE / MODIFY / REJECT / WAIT | `uvicorn app.main:app` |
| **`research/`, `research_data/`** | The chronological 70/30 experiment that tests whether the AI layer improves the bot, net of its own API cost | `python -m research.cli`, `python -m research_data` |

The strategy under test is the **real production bot**, ported from
`scalp-sample-v2` and verified bar-for-bar against its own code. Read
[`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) before any result — it states what
the experiment can and cannot support.

---

## Quick start

```bash
git clone https://github.com/mikaeilbarout/trade_agent
cd trade_agent
git checkout claude/multi-agent-trading-ai-n98tuj

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

pytest                      # 551 tests, no network and no API key needed
```

That is the whole install. Everything below is optional.

### Run the live decision service

```bash
cp .env.example .env         # the mock providers work out of the box
uvicorn app.main:app --reload
# → http://localhost:8000/docs
```

Or `docker compose up --build` for the API plus Postgres and Redis.

### Run the research pipeline

```bash
python -m research.cli status        # what exists, what runs next
python -m research_data readiness    # full data readiness report
```

---

## Requirements

* **Python 3.11+** (the code uses `StrEnum`, `zoneinfo` and PEP 604 unions).
* Everything else comes from `requirements.txt`. No database, no Redis and no API
  key are needed to run the tests.
* `tzdata` is pinned because the calendar's release-time conversion needs a
  zoneinfo database, which slim containers and Windows do not ship.

---

## Running the tests

```bash
pytest                                   # all 551
pytest tests/unit tests/integration -q   # the live service only
pytest tests/research -q                 # the experiment only
pytest -q tests/research/test_donchian_strategy.py   # the bot port
```

The suite makes **no network requests and no paid API calls**, and needs no
credentials. An autouse fixture clears `FRED_API_KEY`, `ANTHROPIC_API_KEY` and
`NEWSAPI_API_KEY` and disables `.env` loading for every test, so results do not
depend on what happens to be configured on the machine.

| Directory | Covers |
|---|---|
| `tests/unit/` | risk engine, decision policy, indicators, data services, agents in isolation |
| `tests/integration/` | the full agent chain, the API contract, all four trading modes, API-key auth |
| `tests/failure/` | provider outages, malformed LLM output, each chain link failing in turn |
| `tests/backtest/` | outcome simulation, look-ahead guards |
| `tests/research/` | the experiment: execution bridge, counterfactuals, A/B config and manifest equality, development-only optimizer, seal enforcement, live↔research policy parity, dataset provenance, cost report, both CLIs, checkpoint/resume, **and the differential test that proves the bot port is faithful** |

Three tests in `test_donchian_strategy.py` compare against the production bot's
own source. They **skip** (loudly, with a reason) unless `scalp-sample-v2` is
cloned at `/home/user/scalp-sample-v2`; the other 548 always run.

---

## Configuration

Every setting is an environment variable read from `.env`. **`.env` is
gitignored and must never be committed** — `.env.example` documents every
variable with its default and is the file to copy.

Nothing below is required for the tests. Grouped by what it unlocks:

### Runs with no configuration at all
`TRADING_MODE=paper`, `LLM_PROVIDER=mock`, `NEWS_PROVIDER=mock`,
`SENTIMENT_PROVIDER=mock`, `MARKET_DATA_PROVIDER=mock`, `DATABASE_URL`
(defaults to SQLite in tests).

### Needed for real AI decisions
| Variable | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` | the only credential the AI layer needs |
| `AI_TECHNICAL_MODEL`, `AI_NEWS_MODEL`, `AI_SENTIMENT_MODEL` | the cheap analyst models (default `claude-haiku-4-5`) |
| `AI_FINAL_MODEL` | the adjudicator (default `claude-sonnet-5`) |
| `AI_COST_LIMIT_USD` | hard spend cap; the run stops safely when reached |

### Needed for historical data ingestion
| Variable | Purpose |
|---|---|
| `FRED_API_KEY` | economic calendar. **Free** — register at <https://fredaccount.stlouisfed.org/apikeys>, no card. The only credential ingestion needs; GDELT requires none. |
| `INGEST_RELEASE_TIME_POLICY` | `SCHEDULED_LOCAL` (current) or `END_OF_DAY` |

Ingestion fetches from exactly two hosts, both of which must be reachable:

```
data.gdeltproject.org     news + point-in-time sentiment   (no key)
api.stlouisfed.org        economic calendar vintages       (needs FRED_API_KEY)
```

### Risk limits and decision thresholds
`MAX_RISK_PER_TRADE_PCT`, `MAX_DAILY_LOSS_PCT`, `MIN_CONFIDENCE`,
`MIN_WEIGHTED_SCORE`, `VETO_ON_*_BLOCK`, `WEIGHT_*` and the rest are in
`.env.example`. For the **experiment** these are set once in
`research/experiment.py::ExperimentConfig` and applied to both arms, so the live
service and the backtest cannot drift apart.

---

## The strategy under test

The signal logic is the production bot, ported from `scalp-sample-v2`
(`mt5/profiles/profile_m15.py` + `strategy/donchian.py`) into
`research/strategy/donchian_scalp.py`:

* Donchian(10) breakout, excluding the current bar
* H4 EMA(30) trend filter with a 0.5% minimum-strength gate
* ATR(14) stops at 3.0×, reward:risk 3.0
* 7-day time stop; 2-hour pause after 3 consecutive losses

ATR is a **simple mean of True Range**, not Wilder's, because that is what the
bot computes and every stop distance depends on it.

### One deliberate correction to the bot's backtest

The bot's backtest aligned the H4 trend with `merge_asof(..., on="ts")`, where
`ts` is the bar's **open** time — so an entry at 16:15 read an H4 bar that does
not close until 20:00, reading up to four hours into the future on every bar.

The bot's **live** code does not do this: it drops the forming bar so
`.iloc[-1]` is a closed one. Live is correct; the backtest was not. Measured on
the supplied data: **7,044 signals with the look-ahead, 5,631 without** — 1,586
(22.5%) existed only because the backtest could see the future.

This port defaults to the live, leakage-free alignment.
`trend_alignment="legacy_open_bar"` reproduces the biased version **for the
differential test only** and is unreachable from `ExperimentConfig`.

`research/strategy/seventy_thirty.py` is the superseded placeholder that stood in
before the bot was available. Nothing in the experiment path imports it.

---

## Market data

`data/source/xauusd_m15_5y.csv` (committed) is the supplied XAUUSD export:
**100,000 M15 bars, 2022-06-21 → 2026-09-11**, sha256 `cc14d9164f64c271…`. No
duplicates, monotonic, OHLC internally consistent, weekday-only.

`data/candles/XAUUSD_M15.parquet` and `XAUUSD_M240.parquet` (also committed) are
the imported bars and the derived H4 trend feed, so a clone runs immediately.
Regenerate either from the source with:

```bash
python -m research_data import-candles --csv data/source/xauusd_m15_5y.csv
```

Two limits worth knowing:

* It is **candles, not ticks**, so intrabar sequencing cannot be resolved. When a
  bar contains both the stop and the target, the engine takes the **stop**.
* It carries **no bid/ask**, so the spread is a **model**
  (`CostModel.fallback_spread_price`), not a measurement. The importer records
  that rather than synthesising quotes from the mid.

---

## The 70/30 experiment

```bash
# historical data ingestion (automated; no CSV to prepare)
python -m research_data sources             # sources, costs, keys, hosts
python -m research_data fetch-news          # GDELT archives      (resumable)
python -m research_data fetch-calendar      # FRED vintages       (needs the key)
python -m research_data build-sentiment     # point-in-time tone, no model, no cost
python -m research_data validate            # every check; fails closed
python -m research_data readiness           # the readiness report

# the experiment
python -m research.cli import-candles ...   # or research_data import-candles
python -m research.cli split                # boundary report (still locked)
python -m research.cli develop              # search the first 70% ONLY
python -m research.cli freeze               # seal; unlocks out-of-sample
python -m research.cli baseline             # Experiment A
python -m research.cli pilot --count 100 --mock   # rehearse, zero spend
python -m research.cli pilot --count 100          # Experiment B, measured cost
python -m research.cli compare              # A vs B + counterfactuals
```

Ordering is enforced by the artifacts, not by documentation: `baseline` cannot
read out-of-sample data until `freeze` has written a strategy seal, `freeze`
refuses without a `develop` report, and re-freezing after a seal exists is
refused with its own exit code.

Both arms run through **one** executor and are built from **one**
`ExperimentConfig`; `compare` refuses to report if the two run manifests differ
outside the AI-specific fields.

### Current state

| | Status |
|---|---|
| Market data | **ready** — 100,000 bars; split forms (70,000 dev / 30,000 out-of-sample, 200-bar embargo) |
| Strategy | **ready** — real bot, verified against its own source |
| News / calendar / sentiment | **UNAVAILABLE** — both provider hosts are blocked by the current environment's egress policy |
| Strategy seal | **not written** — out-of-sample access raises `LeakageError` |
| Pilot / backtest | **not run** |

Run `python -m research_data readiness` for the full report.

---

## What this system will not do

* It will not approve a trade the deterministic risk engine refuses. The engine
  runs twice — before any LLM call, and again on the exact trade about to execute.
* It will not treat missing data as neutral data. An agent whose point-in-time
  dataset has no coverage answers `UNAVAILABLE` and the fail-closed policy applies.
* It will not serve a revised economic figure as the value that was public at the
  original release.
* It will not present sentiment produced today as point-in-time evidence.
* It will not show an agent information that became available after the signal.

And one it **cannot** do: point-in-time input data does not give point-in-time
model knowledge. A model shown a 2023 setup may recognise the period from its
training data. That limitation is carried in every manifest and report, and
`docs/METHODOLOGY.md` states why prompt instructions do not fix it.

---

## Layout

```
app/            live FastAPI decision service (agents, risk engine, API, DB)
research/
  strategy/     donchian_scalp.py  ← the real bot;  optimizer.py  ← dev-only search
  backtest/     engine.py (fills), executor.py (the A/B execution bridge)
  data/         split.py (70/30 guard), pit.py (point-in-time queries),
                candle_import.py, ingest/ (GDELT + FRED), readiness.py
  report/       cost, counterfactual, comparison, limitations
  experiment.py one config for both arms + the equality assertion
  cli.py        the experiment CLI
research_data/  the ingestion CLI
tests/          551 tests
docs/           METHODOLOGY.md — read before the numbers
```

---

## Security

* `.env` is gitignored; `.env.example` carries no real values.
* Credentials are never written to logs, errors, reports or the resumable
  checkpoint. FRED passes its key in the query string, so
  `research/data/ingest/redact.py` masks credential parameters at every
  boundary that turns a URL into text, and six regression tests cover it.
* The AI agents never receive broker credentials, database credentials or API
  keys; the execution layer is isolated from them.
