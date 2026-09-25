"""Synthetic API response-time time series generator.

Generates 5-minutely latency data over ~60 days with:
  - daily seasonality (higher latency during business hours)
  - weekly seasonality (quieter weekends)
  - random noise (lognormal)
  - injected incident windows (30-180 min spikes), some severe enough
    to push the rolling p99 over the SLA threshold

Output CSV columns: timestamp, latency_ms, is_incident (ground truth 0/1)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
DAYS = 60
FREQ = "5min"
START = "2026-07-27 00:00:00"
SLA_P99_MS = 800.0


def _daily_profile(hour: np.ndarray) -> np.ndarray:
    """Business-hours hump: peaks ~14:00, quiet overnight."""
    return 220.0 * np.exp(-0.5 * ((hour - 14.0) / 4.0) ** 2)


def _incident_shape(n: int, rng: np.random.Generator) -> np.ndarray:
    """Ramp up fast, jittery plateau, decay tail."""
    ramp = max(1, n // 6)
    shape = np.ones(n)
    shape[:ramp] = np.linspace(0.2, 1.0, ramp)
    shape[ramp:] *= np.linspace(1.0, 0.35, n - ramp)
    shape *= rng.uniform(0.85, 1.15, n)
    return shape


def generate(
    seed: int = SEED,
    days: int = DAYS,
    n_incidents: int = 18,
    out_path: str | Path = "data/latency.csv",
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    idx = pd.date_range(START, periods=days * 24 * 12, freq=FREQ)
    n = len(idx)

    hour = idx.hour + idx.minute / 60.0
    base = 110.0 + _daily_profile(hour.to_numpy())
    weekend = idx.dayofweek >= 5
    base = np.where(weekend, base * 0.75, base)

    noise = rng.lognormal(mean=0.0, sigma=0.18, size=n)
    latency = base * noise

    is_incident = np.zeros(n, dtype=int)

    # Keep incidents away from the very start/end so rolling windows are valid.
    lo, hi = 300, n - 300
    starts = rng.choice(np.arange(lo, hi), size=n_incidents, replace=False)
    starts.sort()

    for s in starts:
        dur = int(rng.integers(6, 37))  # 30 - 180 minutes in 5-min steps
        e = min(s + dur, n)
        severe = rng.random() < 0.6
        amp = rng.uniform(700, 1600) if severe else rng.uniform(150, 450)
        latency[s:e] += amp * _incident_shape(e - s, rng)
        is_incident[s:e] = 1

    df = pd.DataFrame(
        {
            "timestamp": idx,
            "latency_ms": np.round(latency, 2),
            "is_incident": is_incident,
        }
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/latency.csv")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    df = generate(seed=args.seed, out_path=args.out)
    print(f"wrote {args.out}: {len(df)} rows, "
          f"{df['is_incident'].sum()} incident intervals, "
          f"p99(latency)={df['latency_ms'].quantile(0.99):.1f} ms")


if __name__ == "__main__":
    main()
