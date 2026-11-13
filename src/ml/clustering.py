import os, sys, json
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, avg, count, round as rnd
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator


def main():
    spark = SparkSession.builder.appName("KMeansSegmentation").master("local[*]").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(base, "data", "raw", "online_shoppers_intention.csv")
    out_dir = os.path.join(base, "data", "processed")
    os.makedirs(out_dir, exist_ok=True)

    df = spark.read.option("header", "true").option("inferSchema", "true").csv(data_path)
    df = df.withColumn("Revenue_num", when(col("Revenue") == True, 1.0).otherwise(0.0))

    # NOTE: Revenue (label) ko features me NAHI daala. Yehi unsupervised hai.
    features = ["Administrative", "Administrative_Duration", "Informational",
                "Informational_Duration", "ProductRelated", "ProductRelated_Duration",
                "BounceRates", "ExitRates", "PageValues"]

    prep = Pipeline(stages=[
        VectorAssembler(inputCols=features, outputCol="raw_features", handleInvalid="keep"),
        StandardScaler(inputCol="raw_features", outputCol="features", withStd=True, withMean=True),
    ]).fit(df)
    data = prep.transform(df)

    # Best K dhundo (Silhouette score se)
    evaluator = ClusteringEvaluator(featuresCol="features", metricName="silhouette")
    best_k, best_score = 2, -1
    for k in range(2, 8):
        m = KMeans(k=k, seed=42, featuresCol="features").fit(data)
        score = evaluator.evaluate(m.transform(data))
        print(f"K={k} | Silhouette={score:.4f}")
        if score > best_score:
            best_k, best_score = k, score
    print(f"\nBest K = {best_k} (Silhouette = {best_score:.4f})")

    model = KMeans(k=best_k, seed=42, featuresCol="features").fit(data)
    clustered = model.transform(data).withColumnRenamed("prediction", "cluster")

    # Har cluster ka profile (yahan Revenue sirf samajhne ke liye dekha, training me nahi use hua)
    profile = (clustered.groupBy("cluster")
        .agg(count("*").alias("sessions"),
             rnd(avg("PageValues"), 2).alias("avg_page_value"),
             rnd(avg("ProductRelated_Duration"), 1).alias("avg_product_secs"),
             rnd(avg("BounceRates"), 4).alias("avg_bounce"),
             rnd(avg("ExitRates"), 4).alias("avg_exit"),
             rnd(avg("Revenue_num") * 100, 2).alias("conversion_pct"))
        .orderBy("cluster"))
    profile.show(truncate=False)

    profile.toPandas().to_csv(os.path.join(out_dir, "cluster_profile.csv"), index=False)
    with open(os.path.join(out_dir, "cluster_metrics.json"), "w") as f:
        json.dump({"algorithm": "KMeans", "best_k": best_k,
                   "silhouette": round(best_score, 4)}, f, indent=4)
    print("Saved cluster_profile.csv and cluster_metrics.json")
    spark.stop()


if __name__ == "__main__":
    main()