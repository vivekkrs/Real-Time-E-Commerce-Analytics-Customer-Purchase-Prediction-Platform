import os, sys
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
from pyspark.sql import SparkSession


def main():
    spark = SparkSession.builder.appName("EcommerceSQL").master("local[*]").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path = os.path.join(base, "data", "raw", "online_shoppers_intention.csv")
    out = os.path.join(base, "data", "processed")
    os.makedirs(out, exist_ok=True)

    df = (spark.read.option("header", "true").option("inferSchema", "true").csv(path))
    df.printSchema()
    df.createOrReplaceTempView("sessions")

    # Query 1: month-wise conversion
    month_df = spark.sql("""
        SELECT Month, COUNT(*) AS total_sessions,
               SUM(CASE WHEN Revenue = true THEN 1 ELSE 0 END) AS conversions,
               ROUND(SUM(CASE WHEN Revenue = true THEN 1 ELSE 0 END)*100.0/COUNT(*), 2) AS conversion_rate_pct,
               ROUND(AVG(PageValues), 2) AS avg_page_value,
               ROUND(AVG(ExitRates)*100.0, 2) AS avg_exit_rate_pct
        FROM sessions GROUP BY Month ORDER BY total_sessions DESC
    """)
    month_df.show(truncate=False)

    # Query 2: visitor type
    spark.sql("""
        SELECT VisitorType, COUNT(*) AS total_sessions,
               ROUND(SUM(CASE WHEN Revenue = true THEN 1 ELSE 0 END)*100.0/COUNT(*), 2) AS conversion_rate_pct,
               ROUND(AVG(ProductRelated_Duration), 2) AS avg_product_seconds
        FROM sessions GROUP BY VisitorType
    """).show(truncate=False)

    # Query 3: traffic source ranking with window function
    spark.sql("""
        SELECT TrafficType, sessions, conversions,
               DENSE_RANK() OVER (ORDER BY conversions DESC) AS sales_volume_rank
        FROM (SELECT TrafficType, COUNT(*) AS sessions,
                     SUM(CASE WHEN Revenue = true THEN 1 ELSE 0 END) AS conversions
              FROM sessions GROUP BY TrafficType)
        ORDER BY sales_volume_rank
    """).show()

    # Query 4: PageValues vs purchase
    spark.sql("""
        SELECT CASE WHEN PageValues > 0 THEN 'Has PageValue' ELSE 'Zero PageValue' END AS segment,
               COUNT(*) AS sessions,
               ROUND(SUM(CASE WHEN Revenue = true THEN 1 ELSE 0 END)*100.0/COUNT(*), 2) AS conversion_rate_pct
        FROM sessions GROUP BY 1
    """).show()

    # Save CSV
    month_df.toPandas().to_csv(os.path.join(out, "batch_monthly_summary.csv"), index=False)
    print("Saved batch_monthly_summary.csv")
    spark.stop()


if __name__ == "__main__":
    main()