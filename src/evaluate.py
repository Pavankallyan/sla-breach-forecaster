"""Evaluate raised alerts against ground-truth SLA breach intervals.

Ground truth: actual 1-hour rolling p99 latency > SLA threshold.
Predictions: alert intervals from the alerting module.

Reports interval-level precision / recall / F1 / false-alarm rate plus
event-level detection stats (how many breach events were caught and the
mean detection delay).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SLA_P99_MS = 800.0
P99_WINDOW = 12  # 1h of 5-min intervals


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous (start, end-exclusive) runs of True."""
    runs, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def evaluate(
    data_path: str | Path = "data/latency.csv",
    alerts_path: str | Path = "data/alerts.csv",
    threshold: float = SLA_P99_MS,
) -> dict:
    df = pd.read_csv(data_path, parse_dates=["timestamp"]).sort_values("timestamp")
    df = df.reset_index(drop=True)

    actual_p99 = df["latency_ms"].rolling(P99_WINDOW, min_periods=1).quantile(0.99)
    breach = (actual_p99 > threshold).to_numpy()

    pred = np.zeros(len(df), dtype=bool)
    ts = df["timestamp"]
    alerts = pd.read_csv(alerts_path, parse_dates=["start_time", "end_time"]) \
        if Path(alerts_path).exists() else pd.DataFrame()
    for _, a in alerts.iterrows():
        pred |= (ts >= a["start_time"]).to_numpy() & (ts <= a["end_time"]).to_numpy()

    tp = int((pred & breach).sum())
    fp = int((pred & ~breach).sum())
    fn = int(((~pred) & breach).sum())
    tn = int(((~pred) & ~breach).sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    far = fp / (fp + tn) if fp + tn else 0.0

    # Event-level: each contiguous ground-truth breach run is one event.
    events = _runs(breach)
    detected, delays = 0, []
    for s, e in events:
        hits = np.where(pred[s:e])[0]
        if len(hits):
            detected += 1
            delays.append(int(hits[0]))  # intervals from event start to alert

    report = {
        "intervals_total": len(df),
        "breach_intervals": int(breach.sum()),
        "alert_intervals": int(pred.sum()),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_alarm_rate": far,
        "breach_events": len(events),
        "events_detected": detected,
        "event_detection_rate": detected / len(events) if events else 0.0,
        "mean_detection_delay_intervals": float(np.mean(delays)) if delays else None,
        "n_alerts": len(alerts),
    }
    return report


def print_report(r: dict) -> None:
    print("=" * 52)
    print("SLA BREACH FORECASTER - EVALUATION REPORT")
    print("=" * 52)
    print(f"Intervals evaluated      : {r['intervals_total']}")
    print(f"Ground-truth breaches    : {r['breach_intervals']} intervals "
          f"in {r['breach_events']} events")
    print(f"Alerts raised            : {r['n_alerts']} "
          f"({r['alert_intervals']} intervals)")
    print("-" * 52)
    print(f"True positives           : {r['tp']}")
    print(f"False positives          : {r['fp']}")
    print(f"False negatives          : {r['fn']}")
    print(f"Precision                : {r['precision']:.3f}")
    print(f"Recall                   : {r['recall']:.3f}")
    print(f"F1 score                 : {r['f1']:.3f}")
    print(f"False-alarm rate         : {r['false_alarm_rate']:.4f}")
    print("-" * 52)
    print(f"Events detected          : {r['events_detected']}/{r['breach_events']} "
          f"({r['event_detection_rate']:.1%})")
    if r["mean_detection_delay_intervals"] is not None:
        mins = r["mean_detection_delay_intervals"] * 5
        print(f"Mean detection delay     : "
              f"{r['mean_detection_delay_intervals']:.1f} intervals (~{mins:.0f} min)")
    print("=" * 52)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/latency.csv")
    ap.add_argument("--alerts", default="data/alerts.csv")
    args = ap.parse_args()
    print_report(evaluate(args.data, args.alerts))


if __name__ == "__main__":
    main()
