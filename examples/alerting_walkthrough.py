"""End-to-end walkthrough: synthetic incidents -> forecast -> alerts -> evaluation.

Runs the full pipeline on a small, fast scenario (14 days of 5-minute
latency with 4 injected incidents, fixed seed) so it finishes in seconds.
All intermediate CSVs go to a temp directory — nothing under ``data/``
is touched.

Usage:
    python examples/alerting_walkthrough.py

Expected outcome (seed 7): all 4 incident events raise exactly one alert
each, with zero false positives — precision 1.000, recall 0.741, and a
mean detection delay of ~5 minutes (one 5-min interval).
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.generate_latency import generate
from src.forecast import forecast
from src.alerts import raise_alerts
from src.evaluate import evaluate, print_report

SEED = 7
DAYS = 14
N_INCIDENTS = 4


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="sla_walkthrough_"))

    print(f"[1/4] generating {DAYS} days of synthetic latency "
          f"({N_INCIDENTS} injected incidents, seed {SEED})...")
    df = generate(seed=SEED, days=DAYS, n_incidents=N_INCIDENTS,
                  out_path=tmp / "latency.csv")
    print(f"      {len(df)} intervals, "
          f"{int(df['is_incident'].sum())} incident intervals")

    print("[2/4] fitting Holt-Winters and building the p99 envelope...")
    fc = forecast(data_path=tmp / "latency.csv",
                  out_path=tmp / "forecast.csv")

    print("[3/4] raising breach alerts (800 ms SLA, 3-interval persistence, "
          "1h cooldown)...")
    alerts = raise_alerts(fc["yhat_p99"], fc["timestamp"])
    alerts.to_csv(tmp / "alerts.csv", index=False)
    for i, a in alerts.iterrows():
        print(f"      alert {i + 1}: {a['start_time']} -> {a['end_time']}, "
              f"peak p99 {a['peak_p99_ms']:.1f} ms "
              f"({int(a['n_intervals'])} intervals)")

    print("[4/4] evaluating against the injected ground truth...")
    report = evaluate(data_path=tmp / "latency.csv",
                      alerts_path=tmp / "alerts.csv")
    print_report(report)

    assert report["events_detected"] == N_INCIDENTS, \
        f"expected {N_INCIDENTS} events detected, got {report['events_detected']}"
    assert report["fp"] == 0, f"expected 0 false positives, got {report['fp']}"
    print("\nAll checks passed: every injected incident was caught with "
          "no false alarms.")


if __name__ == "__main__":
    main()
