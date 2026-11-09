"""Unit tests for src/generate_latency.py."""

import numpy as np
import pandas as pd
import pytest

from src.generate_latency import (
    _daily_profile,
    _incident_shape,
    generate,
)


def _gen(tmp_path, **kwargs):
    kwargs.setdefault("seed", 42)
    kwargs.setdefault("days", 7)
    kwargs.setdefault("n_incidents", 3)
    kwargs.setdefault("out_path", tmp_path / "latency.csv")
    return generate(**kwargs)


def test_schema_and_spacing(tmp_path):
    df = _gen(tmp_path)
    assert list(df.columns) == ["timestamp", "latency_ms", "is_incident"]
    assert len(df) == 7 * 24 * 12  # 5-minute intervals over 7 days
    diffs = df["timestamp"].diff().dropna()
    assert (diffs == pd.Timedelta(minutes=5)).all()
    assert df["timestamp"].is_monotonic_increasing
    assert df["latency_ms"].notna().all()
    assert (df["latency_ms"] > 0).all()
    assert set(df["is_incident"].unique()) <= {0, 1}


def test_default_run_known_values(tmp_path):
    df = _gen(tmp_path)
    assert int(df["is_incident"].sum()) == 57
    assert df["latency_ms"].quantile(0.99) == pytest.approx(1030.1, abs=0.5)
    first = int(np.where(df["is_incident"].to_numpy() == 1)[0][0])
    last = int(np.where(df["is_incident"].to_numpy() == 1)[0][-1])
    assert first == 364 and last == 1431


def test_deterministic_same_seed(tmp_path):
    a = _gen(tmp_path)
    b = _gen(tmp_path)
    pd.testing.assert_frame_equal(a, b)


def test_different_seed_differs(tmp_path):
    a = _gen(tmp_path, seed=42)
    b = _gen(tmp_path, seed=7)
    assert not a.equals(b)


def test_no_incidents(tmp_path):
    df = _gen(tmp_path, n_incidents=0)
    assert int(df["is_incident"].sum()) == 0
    # Baseline only: p99 stays far below the 800 ms SLA threshold.
    assert df["latency_ms"].quantile(0.99) < 800.0


def test_incidents_away_from_edges(tmp_path):
    df = _gen(tmp_path)
    assert df["is_incident"].iloc[:300].sum() == 0
    assert df["is_incident"].iloc[-300:].sum() == 0


def test_weekend_quieter_than_weekday(tmp_path):
    df = _gen(tmp_path, n_incidents=0)
    weekday = df[df["timestamp"].dt.dayofweek < 5]["latency_ms"].mean()
    weekend = df[df["timestamp"].dt.dayofweek >= 5]["latency_ms"].mean()
    assert weekend < weekday


def test_writes_csv_matching_return(tmp_path):
    out = tmp_path / "sub" / "latency.csv"
    df = _gen(tmp_path, out_path=out)
    assert out.exists()
    pd.testing.assert_frame_equal(df, pd.read_csv(out, parse_dates=["timestamp"]))


def test_daily_profile_peaks_at_14h():
    hours = np.arange(0, 24, dtype=float)
    profile = _daily_profile(hours)
    assert profile[14] == pytest.approx(220.0)
    assert profile[14] == profile.max()
    assert profile[3] == pytest.approx(5.0, abs=0.5)
    assert (profile > 0).all()


def test_incident_shape_ramps_then_decays():
    rng = np.random.default_rng(0)
    shape = _incident_shape(12, rng)
    assert len(shape) == 12
    assert shape[0] == pytest.approx(0.2, abs=0.05)  # starts low
    assert shape.max() <= 1.15 and shape.min() >= 0.15
    # Ramp portion rises; the tail decays below the plateau.
    assert shape[1] > shape[0]
    assert shape[-1] < shape[2]
