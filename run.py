"""End-to-end pipeline: generate -> forecast -> alert -> evaluate -> plots.

Usage:
    python run.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from generate_latency import generate  # noqa: E402
from forecast import forecast  # noqa: E402
from alerts import raise_alerts  # noqa: E402
from evaluate import evaluate, print_report  # noqa: E402
from plot import (  # noqa: E402
    plot_full_series,
    plot_forecast,
    plot_alerts,
)


def main() -> None:
    data_dir = ROOT / "data"
    plots_dir = ROOT / "plots"
    data_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)

    print("[1/5] generating synthetic latency series...")
    df = generate(out_path=data_dir / "latency.csv")

    print("[2/5] fitting Holt-Winters model and forecasting 24h...")
    fc = forecast(
        data_path=data_dir / "latency.csv",
        out_path=data_dir / "forecast.csv",
    )

    print("[3/5] raising breach alerts...")
    alerts = raise_alerts(fc["yhat_p99"], fc["timestamp"])
    alerts.to_csv(data_dir / "alerts.csv", index=False)
    print(f"      {len(alerts)} alerts raised")

    print("[4/5] evaluating...")
    report = evaluate(
        data_path=data_dir / "latency.csv",
        alerts_path=data_dir / "alerts.csv",
    )
    print_report(report)

    print("[5/5] plotting...")
    plot_full_series(df, plots_dir / "01_full_series.png")
    plot_forecast(df, fc, plots_dir / "02_forecast_vs_actual.png")
    plot_alerts(df, fc, alerts, plots_dir / "03_alerts.png")
    print("      saved plots to plots/")

    print("\nDone. Artifacts: data/latency.csv, data/forecast.csv, "
          "data/alerts.csv, plots/*.png")


if __name__ == "__main__":
    main()
