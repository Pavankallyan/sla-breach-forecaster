"""Forecasting model: Holt-Winters (additive, daily seasonality).

Trains on all but the last 24h of the synthetic series, forecasts the next
24h, and builds an upper p99 envelope:

    predicted_p99(t) = point_forecast(t) + q99(training residuals)

For the historical (training) portion, in-sample one-step-ahead fitted
values are used so the full series has a model-based p99 estimate that the
alerting module can consume.

Output: data/forecast.csv with columns timestamp, yhat, yhat_p99
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

TEST_HOURS = 24
SEASONAL_PERIODS = 288  # 24h / 5min
RESID_QUANTILE = 0.99


def forecast(
    data_path: str | Path = "data/latency.csv",
    out_path: str | Path = "data/forecast.csv",
) -> pd.DataFrame:
    df = pd.read_csv(data_path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    test_n = TEST_HOURS * 12
    train = df.iloc[:-test_n]
    test = df.iloc[-test_n:]

    model = ExponentialSmoothing(
        train["latency_ms"],
        trend="add",
        damped_trend=True,
        seasonal="add",
        seasonal_periods=SEASONAL_PERIODS,
    )
    fit = model.fit(optimized=True, use_brute=False)

    fitted = fit.fittedvalues
    fc = fit.forecast(test_n)

    residuals = (train["latency_ms"].to_numpy() - np.asarray(fitted)).astype(float)
    residuals = residuals[np.isfinite(residuals)]
    q = float(np.quantile(residuals, RESID_QUANTILE))

    yhat = pd.Series(np.concatenate([np.asarray(fitted), np.asarray(fc)]),
                     index=df.index, dtype=float)
    out = pd.DataFrame(
        {
            "timestamp": df["timestamp"],
            "yhat": yhat.round(2),
            "yhat_p99": (yhat + q).round(2),
        }
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"train={len(train)} test={len(test)} "
          f"residual q99={q:.1f} ms (added to point forecast for p99 envelope)")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/latency.csv")
    ap.add_argument("--out", default="data/forecast.csv")
    args = ap.parse_args()
    forecast(args.data, args.out)


if __name__ == "__main__":
    main()
