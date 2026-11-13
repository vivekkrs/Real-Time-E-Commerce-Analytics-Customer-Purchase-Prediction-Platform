import csv
import json
from pathlib import Path

import pytest

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