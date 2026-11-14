import os, sys, time, json
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.sql import SparkSession
from pyspark.sql.types import (StructType, StructField, StringType,
                               IntegerType, DoubleType, BooleanType)
from pyspark.sql.functions import col, when
from pyspark.ml.functions import vector_to_array
from pyspark.ml import PipelineModel


def main():
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_dir = os.path.join(base, "models", "spark_purchase_model")
    buffer_file = os.path.join(base, "data", "processed", "live_stream_buffer.jsonl")
    pred_file = os.path.join(base, "data", "processed", "live_predictions.jsonl")

    if not os.path.exists(model_dir):
        sys.exit("Model nahi mila. Pehle: python src/ml/train.py")

    spark = SparkSession.builder.appName("RealtimeScoring").master("local[*]").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    model = PipelineModel.load(model_dir)

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

    processed = 0
    try:
        while True:
            if not os.path.exists(buffer_file):
                time.sleep(1); continue
            with open(buffer_file, encoding="utf-8") as f:
                lines = f.readlines()
            batch = lines[processed:processed + 30]
            if not batch:
                time.sleep(1); continue

            records = [json.loads(l) for l in batch if l.strip()]
            df = spark.createDataFrame(records, schema=schema)
            df = df.withColumn("Weekend_num", when(col("Weekend") == True, 1.0).otherwise(0.0))

            scored = (model.transform(df)
                      .withColumn("purchase_prob", vector_to_array(col("probability"))[1]))
            rows = scored.select("session_id", "event_timestamp", "VisitorType", "Month",
                                 "PageValues", "ExitRates", "prediction",
                                 "purchase_prob", "Revenue").collect()

            high = 0
            with open(pred_file, "a", encoding="utf-8") as pf:
                for r in rows:
                    d = r.asDict()
                    high += d["prediction"] == 1.0
                    pf.write(json.dumps(d) + "\n")

            processed += len(batch)
            print(f"[BATCH] Scored {len(rows)} | High-intent: {high} | Total: {processed}")
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()