"""Unit tests for src/evaluate.py.

Ground truth in these tests is a hand-built latency series: 200 five-minute
intervals at a 100 ms baseline with one 30-interval block at 2000 ms
(indices 50..79). With the 12-interval rolling p99 window, pandas'
quantile interpolation means a single high sample already pushes the
window p99 to ~1791 ms, so the ground-truth breach mask is indices
50..90: 41 intervals in a single event.
"""

import numpy as np
import pandas as pd
import pytest

from src.evaluate import _runs, evaluate, print_report

N = 200
BREACH_START = 50
BREACH_END = 90  # inclusive; 41 breach intervals, one event


def _latency_df():
    ts = pd.date_range("2026-07-27", periods=N, freq="5min")
    latency = np.full(N, 100.0)
    latency[50:80] = 2000.0
    return pd.DataFrame({"timestamp": ts, "latency_ms": latency})


def _write_latency(tmp_path):
    path = tmp_path / "latency.csv"
    _latency_df().to_csv(path, index=False)
    return path


def _write_alerts(tmp_path, intervals):
    """intervals: list of (start_idx, end_idx) into the latency series."""
    ts = pd.date_range("2026-07-27", periods=N, freq="5min")
    rows = [
        {"start_time": ts[s], "end_time": ts[e], "peak_p99_ms": 2000.0,
         "n_intervals": e - s + 1}
        for s, e in intervals
    ]
    path = tmp_path / "alerts.csv"
    pd.DataFrame(rows, columns=["start_time", "end_time", "peak_p99_ms",
                                "n_intervals"]).to_csv(path, index=False)
    return path


# --- _runs ---------------------------------------------------------------

def test_runs_empty():
    assert _runs(np.array([], dtype=bool)) == []


def test_runs_all_false():
    assert _runs(np.zeros(5, dtype=bool)) == []


def test_runs_all_true():
    assert _runs(np.ones(5, dtype=bool)) == [(0, 5)]


def test_runs_single_and_multiple():
    mask = np.array([False, True, True, False, True, False, True, True, True])
    assert _runs(mask) == [(1, 3), (4, 5), (6, 9)]


# --- evaluate: ground truth ----------------------------------------------

def test_breach_intervals_and_events(tmp_path):
    r = evaluate(_write_latency(tmp_path), tmp_path / "missing_alerts.csv")
    assert r["intervals_total"] == N
    assert r["breach_intervals"] == 41
    assert r["breach_events"] == 1


def test_perfect_alert(tmp_path):
    data = _write_latency(tmp_path)
    alerts = _write_alerts(tmp_path, [(BREACH_START, BREACH_END)])
    r = evaluate(data, alerts)
    assert r["tp"] == 41 and r["fp"] == 0 and r["fn"] == 0
    assert r["tp"] + r["fp"] + r["fn"] + r["tn"] == N
    assert r["precision"] == pytest.approx(1.0)
    assert r["recall"] == pytest.approx(1.0)
    assert r["f1"] == pytest.approx(1.0)
    assert r["false_alarm_rate"] == pytest.approx(0.0)
    assert r["events_detected"] == 1
    assert r["event_detection_rate"] == pytest.approx(1.0)
    assert r["mean_detection_delay_intervals"] == 0
    assert r["n_alerts"] == 1


def test_no_alerts_file(tmp_path):
    r = evaluate(_write_latency(tmp_path), tmp_path / "missing_alerts.csv")
    assert r["n_alerts"] == 0
    assert r["tp"] == 0 and r["fp"] == 0 and r["fn"] == 41
    assert r["tn"] == N - 41
    assert r["precision"] == 0.0 and r["recall"] == 0.0 and r["f1"] == 0.0
    assert r["events_detected"] == 0
    assert r["event_detection_rate"] == 0.0
    assert r["mean_detection_delay_intervals"] is None


def test_partial_alert(tmp_path):
    data = _write_latency(tmp_path)
    alerts = _write_alerts(tmp_path, [(55, 60)])  # 6 of the 41 breach intervals
    r = evaluate(data, alerts)
    assert r["tp"] == 6 and r["fp"] == 0 and r["fn"] == 35
    assert r["recall"] == pytest.approx(6 / 41)
    assert r["precision"] == pytest.approx(1.0)
    assert r["events_detected"] == 1
    assert r["mean_detection_delay_intervals"] == 5  # alert starts 5 in


def test_false_alarm_only(tmp_path):
    data = _write_latency(tmp_path)
    alerts = _write_alerts(tmp_path, [(10, 15)])  # baseline only, 6 intervals
    r = evaluate(data, alerts)
    assert r["tp"] == 0 and r["fp"] == 6 and r["fn"] == 41
    assert r["precision"] == 0.0 and r["recall"] == 0.0
    assert r["events_detected"] == 0
    assert r["mean_detection_delay_intervals"] is None


def test_custom_threshold_no_breach(tmp_path):
    # 2500 ms is above every rolling p99 in this series (max 2000 ms).
    r = evaluate(_write_latency(tmp_path), tmp_path / "missing_alerts.csv",
                 threshold=2500.0)
    assert r["breach_intervals"] == 0
    assert r["breach_events"] == 0
    assert r["event_detection_rate"] == 0.0
    assert r["fn"] == 0 and r["tn"] == N


def test_unsorted_input_same_result(tmp_path):
    data = _write_latency(tmp_path)
    alerts = _write_alerts(tmp_path, [(BREACH_START, BREACH_END)])
    shuffled = tmp_path / "shuffled.csv"
    _latency_df().sample(frac=1.0, random_state=7).to_csv(shuffled, index=False)
    assert evaluate(data, alerts) == evaluate(shuffled, alerts)


def test_print_report_runs(tmp_path, capsys):
    r = evaluate(_write_latency(tmp_path), tmp_path / "missing_alerts.csv")
    print_report(r)
    out = capsys.readouterr().out
    assert "Precision" in out and "Events detected" in out
    assert "0/1" in out
