# SLA Breach Forecaster

Predicting API latency SLA violations before (or as) they happen — a small
time-series forecasting + alerting pipeline built on **synthetic** API
response-time data.

## Overview

I built this project to practice the kind of work that sits at the
intersection of my background in software test engineering and what I'm
studying now in my MS in Data Science: take a noisy operational signal (API
latency), model its normal behavior, and raise reliable alerts when it is
about to breach its SLA.

The pipeline:

1. **Generate** 60 days of 5-minutely synthetic API latency with daily and
   weekly seasonality plus injected incident windows (ground truth labeled).
2. **Forecast** the next 24h with Holt-Winters exponential smoothing and
   build a p99 upper envelope from the model's residual distribution.
3. **Alert** when the modelled p99 stays above the SLA threshold
   (800 ms) for 3 consecutive intervals, with a 1-hour cooldown to avoid
   flapping.
4. **Evaluate** alerts against ground-truth breach intervals: precision,
   recall, F1, false-alarm rate, plus event-level detection stats.
5. **Plot** the series, the forecast, and the alerts.

Everything is synthetic — the data generator, the incidents, and the ground
truth labels. There are no real employers, services, or production metrics
involved.

## Approach

- **Data** (`src/generate_latency.py`): 17,280 five-minute samples
  (2026-07-27 → 2026-09-24) with a business-hours latency hump (peaks
  ~14:00), quieter weekends, lognormal noise, and 18 injected incidents
  lasting 30–180 minutes. ~60% of incidents are severe (spike 700–1600 ms,
  enough to push the 1h rolling p99 over the 800 ms SLA); the rest are mild
  (150–450 ms) and should *not* page. Fixed seed (42) for reproducibility.
- **Model** (`src/forecast.py`): additive Holt-Winters
  (`statsmodels.tsa.holtwinters.ExponentialSmoothing`) with a damped trend
  and a 288-step daily seasonal cycle, trained on the first 59 days, 24h
  held out. The p99 envelope is
  `point_forecast + q99(training residuals)` — a simple, explainable way to
  turn a point forecast into an upper-tail estimate.
- **Alerting** (`src/alerts.py`): fires when the modelled p99 exceeds
  800 ms for 3 consecutive intervals; clears when it drops back under; a
  1-hour cooldown after each alert suppresses re-triggering on a signal
  hovering near the threshold.
- **Evaluation** (`src/evaluate.py`): ground truth = actual 1h rolling p99
  > 800 ms. Interval-level precision/recall/F1/false-alarm rate, plus
  event-level detection rate and mean detection delay.

## Results on demo data

From a single `python run.py` on the synthetic dataset (17,280 intervals,
349 ground-truth breach intervals across 12 events, 13 alerts raised):

| Metric | Value |
|---|---|
| Precision | **0.973** |
| Recall | **0.713** |
| F1 score | **0.823** |
| False-alarm rate | **0.0004** |
| Breach events detected | **12 / 12 (100%)** |
| Mean detection delay | **~5 min** (1.1 intervals) |

Reading: every real breach event was caught, typically within one 5-minute
interval, and only 7 of the 256 alerted intervals were false alarms. Recall
is the weaker number — the model-based p99 envelope lags the steepest part
of a spike, so the first few breach intervals of fast-rising incidents are
missed before the alert fires (persistence = 3 intervals also costs a
little). That's the honest trade-off of a detection-oriented design on
unpredictable incidents.

## Project structure

```
sla-breach-forecaster/
├── run.py                 # end-to-end entry point (one command)
├── requirements.txt
├── README.md
├── src/
│   ├── generate_latency.py  # synthetic latency series + incident injection
│   ├── forecast.py          # Holt-Winters model, 24h forecast, p99 envelope
│   ├── alerts.py            # threshold + persistence + cooldown alerting
│   ├── evaluate.py          # precision / recall / F1 / false-alarm report
│   └── plot.py              # matplotlib figures
├── data/                    # generated CSVs (latency, forecast, alerts)
└── plots/
    ├── 01_full_series.png       # 60-day series, incidents shaded
    ├── 02_forecast_vs_actual.png # last 72h incl. 24h holdout
    └── 03_alerts.png            # alerts vs ground-truth breaches
```

## How to run

```bash
# 1. Create and activate a virtual environment (Python 3.10+)
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the whole pipeline with one command
python run.py
```

Each stage also runs standalone:

```bash
python src/generate_latency.py --out data/latency.csv
python src/forecast.py        --data data/latency.csv --out data/forecast.csv
python src/alerts.py          --forecast data/forecast.csv --out data/alerts.csv
python src/evaluate.py        --data data/latency.csv --alerts data/alerts.csv
python src/plot.py            --data data/latency.csv --forecast data/forecast.csv \
                              --alerts data/alerts.csv --outdir plots
```

Runtime is dominated by the Holt-Winters fit (~2–3 minutes on 17k points);
everything else takes seconds.

## Design decisions

- **Why Holt-Winters, not Prophet:** Prophet pulls in heavy dependencies
  (and historically painful installs); additive Holt-Winters with an
  explicit 288-step seasonal period captures the daily business-hours
  pattern directly in statsmodels with no extra stack. One caveat I kept:
  the optimizer emits a `ConvergenceWarning` on this data — the resulting
  fit still tracks the series well (see the forecast plot), but it's a
  reminder that smoothing-parameter optimization on 17k noisy points is
  approximate.
- **Why a residual-based p99 envelope:** a point forecast can't page on
  tail behavior. Adding the 99th percentile of training residuals (146.2 ms
  here) is a transparent, one-line way to convert "expected latency" into
  "worst-case-ish latency" without fitting a second model.
- **Why persistence + cooldown:** paging on a single 5-minute breach would
  flap constantly around the threshold; requiring 3 consecutive breaches
  and a 1-hour cooldown after each alert mirrors how real on-call alerting
  (e.g. Prometheus `for:` + alertmanager grouping) avoids alert storms.
  The cost is a few minutes of detection delay — measured above at ~5 min.
- **Why evaluate on modelled p99, not actual:** alerting on the actual
  rolling p99 would be near-trivially perfect. Using the model's
  one-step-ahead envelope (fitted on history, forecasted on the holdout)
  tests whether the *model* can drive the alerting decision — closer to a
  real forecasting use case.
- **Limitations:** incidents are random by construction, so no forecaster
  can predict them in advance — this pipeline is really *fast detection*
  driven by a forecasting model, not true pre-incident prediction. The
  synthetic seasonality is smooth and well-behaved; real API latency has
  deploys, traffic shifts, and multi-seasonal structure this demo doesn't
  model. The residual q99 is global, so the envelope is slightly too wide
  overnight and slightly too tight at peak hours.
