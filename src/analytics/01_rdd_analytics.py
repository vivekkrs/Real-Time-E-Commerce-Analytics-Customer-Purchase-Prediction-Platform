import os, sys
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
from pyspark import SparkConf, SparkContext


def parse(line):
    p = line.strip().split(",")
    if len(p) < 18:
        return None
    try:
        return {
            "bounce": float(p[6]), "exit": float(p[7]), "page_values": float(p[8]),
            "month": p[10].strip(), "visitor": p[15].strip(),
            "revenue": p[17].strip().upper() == "TRUE",
        }
    except ValueError:
        return None


def main():
    conf = SparkConf().setAppName("EcommerceRDD").setMaster("local[*]")
    sc = SparkContext.getOrCreate(conf=conf)
    sc.setLogLevel("WARN")

    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path = os.path.join(base, "data", "raw", "online_shoppers_intention.csv")

    lines = sc.textFile(path)
    header = lines.first()
    rdd = (lines.filter(lambda l: l != header)
                .map(parse).filter(lambda r: r is not None).cache())

    # 1. Overall conversion
    total = rdd.count()
    buyers = rdd.filter(lambda r: r["revenue"]).count()
    print(f"Total: {total} | Buyers: {buyers} | Conversion: {buyers*100/total:.2f}%")

    # 2. Conversion by visitor type (reduceByKey + mapValues)
    by_visitor = (rdd.map(lambda r: (r["visitor"], (1, 1 if r["revenue"] else 0)))
                     .reduceByKey(lambda a, b: (a[0]+b[0], a[1]+b[1]))
                     .mapValues(lambda v: (v[0], v[1], round(v[1]*100/v[0], 2))))
    for k, v in by_visitor.collect():
        print(k, v)

    # 3. Avg bounce/exit by month
    by_month = (rdd.map(lambda r: (r["month"], (r["bounce"], r["exit"], 1)))
                   .reduceByKey(lambda a, b: (a[0]+b[0], a[1]+b[1], a[2]+b[2]))
                   .mapValues(lambda v: (v[2], round(v[0]/v[2], 4), round(v[1]/v[2], 4)))
                   .sortBy(lambda kv: kv[1][0], ascending=False))
    for k, v in by_month.collect():
        print(k, "sessions/bounce/exit =", v)

    # 4. High-intent sessions
    high = rdd.filter(lambda r: r["page_values"] > 50)
    hc = high.count()
    hb = high.filter(lambda r: r["revenue"]).count()
    print(f"High-intent: {hc} | Converted: {hb} ({hb*100/max(hc,1):.2f}%)")

    sc.stop()


if __name__ == "__main__":
    main()