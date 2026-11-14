import os, sys, json
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator


def main():
    spark = SparkSession.builder.appName("MLlibTraining").master("local[*]").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(base, "data", "raw", "online_shoppers_intention.csv")
    model_dir = os.path.join(base, "models", "spark_purchase_model")
    metrics_path = os.path.join(base, "data", "processed", "model_metrics.json")
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

    df = spark.read.option("header", "true").option("inferSchema", "true").csv(data_path)
    df = (df.withColumn("label", when(col("Revenue") == True, 1.0).otherwise(0.0))
            .withColumn("Weekend_num", when(col("Weekend") == True, 1.0).otherwise(0.0)))

    cat_cols = ["Month", "VisitorType"]
    num_cols = ["Administrative", "Administrative_Duration", "Informational",
                "Informational_Duration", "ProductRelated", "ProductRelated_Duration",
                "BounceRates", "ExitRates", "PageValues", "SpecialDay",
                "OperatingSystems", "Browser", "Region", "TrafficType", "Weekend_num"]

    stages = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep") for c in cat_cols]
    stages.append(OneHotEncoder(inputCols=[f"{c}_idx" for c in cat_cols],
                                outputCols=[f"{c}_vec" for c in cat_cols]))
    stages.append(VectorAssembler(inputCols=num_cols + [f"{c}_vec" for c in cat_cols],
                                  outputCol="raw_features", handleInvalid="keep"))
    stages.append(StandardScaler(inputCol="raw_features", outputCol="features",
                                 withStd=True, withMean=False))
    stages.append(RandomForestClassifier(featuresCol="features", labelCol="label",
                                         numTrees=25, maxDepth=6, seed=42))

    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    model = Pipeline(stages=stages).fit(train_df)
    pred = model.transform(test_df)

    roc = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC").evaluate(pred)
    pr = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderPR").evaluate(pred)
    acc = MulticlassClassificationEvaluator(labelCol="label", metricName="accuracy").evaluate(pred)
    f1 = MulticlassClassificationEvaluator(labelCol="label", metricName="f1").evaluate(pred)
    print(f"ROC-AUC={roc:.4f} PR-AUC={pr:.4f} Acc={acc*100:.2f}% F1={f1:.4f}")

    model.write().overwrite().save(model_dir)

    with open(metrics_path, "w") as f:
        json.dump({"model_type": "RandomForestClassifier", "num_trees": 25, "max_depth": 6,
                   "roc_auc": round(roc, 4), "pr_auc": round(pr, 4),
                   "accuracy_pct": round(acc * 100, 2), "f1_score": round(f1, 4),
                   "train_records": train_df.count(), "test_records": test_df.count()}, f, indent=4)
    spark.stop()


if __name__ == "__main__":
    main()
    