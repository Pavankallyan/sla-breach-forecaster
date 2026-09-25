"""Matplotlib plots for the SLA breach forecaster (saved to plots/)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SLA_P99_MS = 800.0


def _shade_runs(ax, ts: pd.Series, mask: np.ndarray, **kw) -> None:
    i, n = 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            ax.axvspan(ts.iloc[i], ts.iloc[j - 1], **kw)
            i = j
        else:
            i += 1


def plot_full_series(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df["timestamp"], df["latency_ms"], lw=0.5, color="#1f77b4", label="latency")
    _shade_runs(ax, df["timestamp"], df["is_incident"].to_numpy() == 1,
                color="red", alpha=0.25)
    ax.axhline(SLA_P99_MS, ls="--", color="red", lw=1, label="SLA p99 = 800 ms")
    ax.set_title("API latency (5-min), 60 days - incident windows shaded")
    ax.set_xlabel("date")
    ax.set_ylabel("latency (ms)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_forecast(df: pd.DataFrame, fc: pd.DataFrame, out: Path) -> None:
    tail = df.tail(72 * 12).copy()
    fct = fc.tail(72 * 12)
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(tail["timestamp"], tail["latency_ms"], lw=0.8, color="#1f77b4",
            label="actual")
    ax.plot(fct["timestamp"], fct["yhat"], lw=0.8, color="orange",
            label="HW forecast (point)")
    ax.plot(fct["timestamp"], fct["yhat_p99"], lw=0.8, color="green",
            label="forecasted p99 envelope")
    ax.axhline(SLA_P99_MS, ls="--", color="red", lw=1, label="SLA p99 = 800 ms")
    ax.set_title("Forecast vs actual - last 72h (final 24h is the holdout)")
    ax.set_xlabel("timestamp")
    ax.set_ylabel("latency (ms)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_alerts(df: pd.DataFrame, fc: pd.DataFrame, alerts: pd.DataFrame,
                out: Path) -> None:
    actual_p99 = df["latency_ms"].rolling(12, min_periods=1).quantile(0.99)
    breach = (actual_p99 > SLA_P99_MS).to_numpy()

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(fc["timestamp"], fc["yhat_p99"], lw=0.6, color="green",
            label="modelled p99")
    ax.plot(df["timestamp"], actual_p99, lw=0.6, color="#1f77b4", alpha=0.7,
            label="actual rolling p99 (1h)")
    _shade_runs(ax, df["timestamp"], breach, color="red", alpha=0.18)
    for _, a in alerts.iterrows():
        ax.axvspan(a["start_time"], a["end_time"], color="orange", alpha=0.35)
    ax.axhline(SLA_P99_MS, ls="--", color="red", lw=1, label="SLA p99 = 800 ms")
    ax.set_title("Breach alerts (orange) vs ground-truth breach intervals (red)")
    ax.set_xlabel("date")
    ax.set_ylabel("p99 latency (ms)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/latency.csv")
    ap.add_argument("--forecast", default="data/forecast.csv")
    ap.add_argument("--alerts", default="data/alerts.csv")
    ap.add_argument("--outdir", default="plots")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.data, parse_dates=["timestamp"])
    fc = pd.read_csv(args.forecast, parse_dates=["timestamp"])
    alerts = pd.read_csv(args.alerts, parse_dates=["start_time", "end_time"]) \
        if Path(args.alerts).exists() else pd.DataFrame()

    plot_full_series(df, outdir / "01_full_series.png")
    plot_forecast(df, fc, outdir / "02_forecast_vs_actual.png")
    plot_alerts(df, fc, alerts, outdir / "03_alerts.png")
    print(f"saved 3 plots to {outdir}/")


if __name__ == "__main__":
    main()
