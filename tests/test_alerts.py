"""Unit tests for src/alerts.py — SLA breach alerting logic.

Covers: persistence requirement, alert window fields, cooldown
suppression, blip handling, custom thresholds, and edge cases.
All tests use small synthetic p99 series so they are fast and
deterministic.
"""

import pandas as pd
import pytest

from src.alerts import raise_alerts

PERSISTENCE = 3
COOLDOWN = 12


def make_series(values, start="2024-01-01 00:00"):
    """Build (p99, timestamps) from a list of p99 values."""
    n = len(values)
    ts = pd.Series(pd.date_range(start, periods=n, freq="5min"))
    return pd.Series(values, dtype=float), ts


def test_no_alerts_when_all_below_threshold():
    p99, ts = make_series([400.0] * 20)
    alerts = raise_alerts(p99, ts, threshold=800.0)
    assert len(alerts) == 0
    assert list(alerts.columns) == ["start_time", "end_time", "peak_p99_ms", "n_intervals"]


def test_empty_series_returns_empty_frame():
    p99, ts = make_series([])
    alerts = raise_alerts(p99, ts)
    assert len(alerts) == 0
    assert list(alerts.columns) == ["start_time", "end_time", "peak_p99_ms", "n_intervals"]


def test_short_blip_not_paged():
    # persistence-1 consecutive intervals above threshold: too short to page on.
    vals = [400.0] * 5 + [850.0] * (PERSISTENCE - 1) + [400.0] * 10
    p99, ts = make_series(vals)
    assert len(raise_alerts(p99, ts)) == 0


def test_exact_persistence_triggers_one_alert():
    vals = [400.0] * 4 + [810.0, 820.0, 830.0] + [400.0] * 10
    p99, ts = make_series(vals)
    alerts = raise_alerts(p99, ts)
    assert len(alerts) == 1
    a = alerts.iloc[0]
    assert a["n_intervals"] == 3
    assert a["start_time"] == ts[4]
    assert a["end_time"] == ts[6]
    assert a["peak_p99_ms"] == 830.0


def test_long_run_single_alert():
    vals = [400.0] * 2 + [900.0] * 7 + [400.0] * 10
    p99, ts = make_series(vals)
    alerts = raise_alerts(p99, ts)
    assert len(alerts) == 1
    assert alerts.iloc[0]["n_intervals"] == 7
    assert alerts.iloc[0]["start_time"] == ts[2]
    assert alerts.iloc[0]["end_time"] == ts[8]


def test_peak_p99_matches_segment_max():
    vals = [400.0] * 3 + [820.0, 950.5, 870.0] + [400.0] * 10
    p99, ts = make_series(vals)
    alerts = raise_alerts(p99, ts)
    assert len(alerts) == 1
    assert alerts.iloc[0]["peak_p99_ms"] == pytest.approx(950.5)


def test_short_blip_does_not_arm_cooldown():
    # A sub-persistence blip must not start a cooldown window: a later
    # genuine run should still page.
    vals = (
        [400.0] * 2
        + [850.0] * (PERSISTENCE - 1)  # blip, ignored
        + [400.0] * 5                  # gap shorter than cooldown
        + [860.0] * 5                  # real run -> alert
        + [400.0] * 20
    )
    p99, ts = make_series(vals)
    alerts = raise_alerts(p99, ts)
    assert len(alerts) == 1
    assert alerts.iloc[0]["n_intervals"] == 5
    assert alerts.iloc[0]["start_time"] == ts[2 + (PERSISTENCE - 1) + 5]


def test_cooldown_suppresses_back_to_back_runs():
    # Second run lands inside the cooldown window after the first alert.
    vals = [900.0] * 5 + [400.0] * 5 + [920.0] * 5 + [400.0] * 20
    p99, ts = make_series(vals)
    alerts = raise_alerts(p99, ts)
    assert len(alerts) == 1
    assert alerts.iloc[0]["start_time"] == ts[0]


def test_second_run_after_cooldown_fires():
    # First run at indices 0-4 -> cooldown until index 5 + COOLDOWN.
    # Second run starts past that -> separate alert.
    start2 = 5 + COOLDOWN + 1
    vals = [900.0] * 5 + [400.0] * (start2 - 5) + [920.0] * 5 + [400.0] * 20
    p99, ts = make_series(vals)
    alerts = raise_alerts(p99, ts)
    assert len(alerts) == 2
    assert alerts.iloc[0]["start_time"] == ts[0]
    assert alerts.iloc[1]["start_time"] == ts[start2]
    assert alerts.iloc[1]["end_time"] == ts[start2 + 4]
    assert alerts.iloc[1]["n_intervals"] == 5


def test_custom_threshold_respected():
    vals = [600.0, 650.0, 620.0] + [400.0] * 10
    p99, ts = make_series(vals)
    assert len(raise_alerts(p99, ts, threshold=500.0)) == 1
    assert len(raise_alerts(p99, ts, threshold=700.0)) == 0


def test_run_to_end_of_series_alerts():
    # Breach run reaching the last interval is still reported.
    vals = [400.0] * 10 + [880.0] * 4
    p99, ts = make_series(vals)
    alerts = raise_alerts(p99, ts)
    assert len(alerts) == 1
    assert alerts.iloc[0]["end_time"] == ts[len(ts) - 1]
    assert alerts.iloc[0]["n_intervals"] == 4
