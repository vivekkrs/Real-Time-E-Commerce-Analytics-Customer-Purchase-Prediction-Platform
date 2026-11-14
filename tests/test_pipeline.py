import csv
import json
import os
from pathlib import Path

import pytest

from dashboard.data_loader import LIVE_COLUMNS, load_live_data
from src.producer.kafka_producer import build_event

ROOT = Path(__file__).resolve().parents[1]


def test_dataset_has_expected_sessions_and_columns():
    with (ROOT / "data/raw/online_shoppers_intention.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 12_331
    assert len(rows[0]) == 18


def test_producer_serializes_dataset_row():
    row = ["0", "0", "0", "0", "1", "0", "0.2", "0.2", "0", "0",
           "Feb", "1", "1", "1", "1", "Returning_Visitor", "FALSE", "TRUE"]
    event = build_event(row)
    assert event["Month"] == "Feb"
    assert event["Weekend"] is False
    assert event["Revenue"] is True
    assert len(event["session_id"]) == 12


def test_producer_rejects_malformed_dataset_row():
    with pytest.raises(ValueError, match="Expected 18 dataset columns"):
        build_event(["Feb"])


def test_model_training_entrypoint_is_available():
    assert (ROOT / "src/ml/train.py").is_file()


def test_metrics_json_has_valid_model_scores():
    with (ROOT / "data/processed/model_metrics.json").open(encoding="utf-8") as f:
        metrics = json.load(f)
    assert metrics["roc_auc"] > 0.85
    assert 0 <= metrics["accuracy_pct"] <= 100


def test_batch_summary_has_monthly_results():
    with (ROOT / "data/processed/batch_monthly_summary.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows
    assert all(0 <= float(row["conversion_rate_pct"]) <= 100 for row in rows)


def test_live_prediction_loader_handles_missing_output(tmp_path):
    result = load_live_data(tmp_path)
    assert result.empty
    assert tuple(result.columns) == LIVE_COLUMNS


def test_live_prediction_loader_reads_latest_files_and_orders_events(tmp_path):
    older = {
        "session_id": "older",
        "event_timestamp": "2026-11-13T10:00:00Z",
        "VisitorType": "Returning_Visitor",
        "Month": "Nov",
        "PageValues": 0.0,
        "ExitRates": 0.1,
        "prediction": 0.0,
        "purchase_prob": 0.2,
        "Revenue": False,
    }
    newer = {**older, "session_id": "newer", "event_timestamp": "2026-11-13T10:01:00Z"}
    (tmp_path / "part-old.json").write_text(json.dumps(older) + "\n", encoding="utf-8")
    (tmp_path / "part-new.json").write_text(json.dumps(newer) + "\n", encoding="utf-8")

    result = load_live_data(tmp_path, max_files=2)
    assert result["session_id"].tolist() == ["older", "newer"]
    assert result["event_timestamp"].dt.tz is not None


def test_live_prediction_loader_limits_files_to_the_newest(tmp_path):
    base = {
        "event_timestamp": "2026-11-13T10:00:00Z",
        "VisitorType": "Returning_Visitor",
        "Month": "Nov",
        "PageValues": 0.0,
        "ExitRates": 0.1,
        "prediction": 0.0,
        "purchase_prob": 0.2,
        "Revenue": False,
    }
    older_path = tmp_path / "part-old.json"
    latest_path = tmp_path / "part-latest.json"
    older_path.write_text(json.dumps({**base, "session_id": "older"}) + "\n", encoding="utf-8")
    latest_path.write_text(json.dumps({**base, "session_id": "latest"}) + "\n", encoding="utf-8")
    older_path.touch()
    latest_path.touch()
    older_path_stat = older_path.stat()
    latest_path_stat = latest_path.stat()
    os.utime(older_path, ns=(older_path_stat.st_atime_ns, latest_path_stat.st_mtime_ns - 1_000_000))

    result = load_live_data(tmp_path, max_files=1)
    assert result["session_id"].tolist() == ["latest"]


def test_live_prediction_loader_rejects_malformed_probability(tmp_path):
    record = {
        "session_id": "bad",
        "event_timestamp": "2026-11-13T10:00:00Z",
        "VisitorType": "Returning_Visitor",
        "Month": "Nov",
        "PageValues": 0.0,
        "ExitRates": 0.1,
        "prediction": 0.0,
        "purchase_prob": 1.2,
        "Revenue": False,
    }
    (tmp_path / "part-bad.json").write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="between 0 and 1"):
        load_live_data(tmp_path)


def test_dashboard_renders():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(ROOT / "dashboard/app.py"), default_timeout=20).run()
    assert not app.exception
    assert [metric.label for metric in app.metric] == [
        "Recent Sessions",
        "High Intent",
        "Intent Rate",
        "Avg. Probability",
    ]