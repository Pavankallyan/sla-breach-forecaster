"""SLA breach alerting.

An alert fires when the (modelled) p99 latency stays above the SLA
threshold for `persistence` consecutive intervals, and clears as soon as
p99 drops back under the threshold. A cooldown window after each alert
prevents flapping when the signal hovers around the threshold.

Output: data/alerts.csv with columns start_time, end_time, peak_p99_ms,
n_intervals
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

SLA_P99_MS = 800.0
PERSISTENCE = 3   # consecutive intervals above threshold to trigger
COOLDOWN = 12     # intervals (1h) to wait after an alert before re-arming


def raise_alerts(
    p99: pd.Series,
    timestamps: pd.Series,
    threshold: float = SLA_P99_MS,
    persistence: int = PERSISTENCE,
    cooldown: int = COOLDOWN,
) -> pd.DataFrame:
    above = (p99 > threshold).to_numpy()
    ts = pd.to_datetime(timestamps).to_numpy()

    alerts: list[dict] = []
    i, n = 0, len(above)
    cooldown_until = -1

    while i < n:
        if i < cooldown_until or not above[i]:
            i += 1
            continue
        # count consecutive breach intervals
        j = i
        while j < n and above[j]:
            j += 1
        run_len = j - i
        if run_len >= persistence:
            seg = p99.iloc[i:j]
            alerts.append(
                {
                    "start_time": pd.Timestamp(ts[i]),
                    "end_time": pd.Timestamp(ts[j - 1]),
                    "peak_p99_ms": round(float(seg.max()), 2),
                    "n_intervals": int(run_len),
                }
            )
            cooldown_until = j + cooldown
            i = j
        else:
            # blip too short to page on; skip past it
            i = j

    return pd.DataFrame(
        alerts,
        columns=["start_time", "end_time", "peak_p99_ms", "n_intervals"],
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--forecast", default="data/forecast.csv")
    ap.add_argument("--out", default="data/alerts.csv")
    ap.add_argument("--threshold", type=float, default=SLA_P99_MS)
    args = ap.parse_args()

    fc = pd.read_csv(args.forecast, parse_dates=["timestamp"])
    alerts = raise_alerts(fc["yhat_p99"], fc["timestamp"], threshold=args.threshold)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    alerts.to_csv(args.out, index=False)
    print(f"raised {len(alerts)} alerts -> {args.out}")
    if len(alerts):
        print(alerts.to_string(index=False))


if __name__ == "__main__":
    main()
