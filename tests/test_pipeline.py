import os, json

def test_dataset_exists():
    assert os.path.exists("data/raw/online_shoppers_intention.csv")

def test_model_saved():
    assert os.path.exists("models/spark_purchase_model")

def test_metrics_json():
    with open("data/processed/model_metrics.json") as f:
        assert json.load(f)["roc_auc"] > 0.85

def test_batch_summary_csv():
    assert os.path.exists("data/processed/batch_monthly_summary.csv")