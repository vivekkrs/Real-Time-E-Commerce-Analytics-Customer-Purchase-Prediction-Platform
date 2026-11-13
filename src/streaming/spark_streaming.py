import os
import sys
import pyspark
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.sql import SparkSession
from pyspark.sql.types import (StructType, StructField, StringType,
                               IntegerType, DoubleType, BooleanType)
from pyspark.sql.functions import col, from_json, when
from pyspark.ml.functions import vector_to_array
from pyspark.ml import PipelineModel


def main():
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_dir = os.path.join(base, "models", "spark_purchase_model")
    pred_dir = os.path.join(base, "data", "processed", "live_predictions")
    checkpoint_dir = os.path.join(base, "checkpoints", "live_predictions")
    os.makedirs(os.path.dirname(pred_dir), exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    if not os.path.exists(model_dir):
        sys.exit("Model not found. Train it first with: python src/ml/train.py")

    S, I, D, B = StringType(), IntegerType(), DoubleType(), BooleanType()
    schema = StructType([
        StructField("session_id", S), StructField("event_timestamp", S),
        StructField("Administrative", I), StructField("Administrative_Duration", D),
        StructField("Informational", I), StructField("Informational_Duration", D),
        StructField("ProductRelated", I), StructField("ProductRelated_Duration", D),
        StructField("BounceRates", D), StructField("ExitRates", D),
        StructField("PageValues", D), StructField("SpecialDay", D),
        StructField("Month", S), StructField("OperatingSystems", I),
        StructField("Browser", I), StructField("Region", I), StructField("TrafficType", I),
        StructField("VisitorType", S), StructField("Weekend", B), StructField("Revenue", B),
    ])

    spark = (SparkSession.builder
             .appName("RealtimeScoring")
             .master("local[*]")
             .config(
                 "spark.jars.packages",
                 f"org.apache.spark:spark-sql-kafka-0-10_2.12:{pyspark.__version__}",
             )
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    try:
        model = PipelineModel.load(model_dir)
        kafka_servers = os.environ.get("KAFKA_BOOTSTRAP", "localhost:9092")
        topic = os.environ.get("KAFKA_TOPIC", "ecommerce-sessions")

        kafka_stream = (spark.readStream
                        .format("kafka")
                        .option("kafka.bootstrap.servers", kafka_servers)
                        .option("subscribe", topic)
                        .option("startingOffsets", "earliest")
                        .load())
        events = (kafka_stream
                  .select(from_json(col("value").cast("string"), schema).alias("event"))
                  .select("event.*")
                  .withColumn("Weekend_num", when(col("Weekend"), 1.0).otherwise(0.0)))

        scored = (model.transform(events)
                  .withColumn("purchase_prob", vector_to_array(col("probability"))[1])
                  .select("session_id", "event_timestamp", "VisitorType", "Month",
                          "PageValues", "ExitRates", "prediction", "purchase_prob", "Revenue"))

        query = (scored.writeStream
                 .format("json")
                 .outputMode("append")
                 .option("path", pred_dir)
                 .option("checkpointLocation", checkpoint_dir)
                 .start())
        query.awaitTermination()
    except KeyboardInterrupt:
        print("Stopping the streaming query.")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()