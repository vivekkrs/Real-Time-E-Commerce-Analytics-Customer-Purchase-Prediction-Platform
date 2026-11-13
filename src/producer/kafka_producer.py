import argparse
import csv
import json
import math
import os
import time
import uuid
from datetime import datetime, timezone


def build_event(row):
    if len(row) != 18:
        raise ValueError(f"Expected 18 dataset columns, got {len(row)}")

    return {
        "session_id": f"SES-{uuid.uuid4().hex[:8].upper()}",
        "event_timestamp": datetime.now(timezone.utc).isoformat(),
        "Administrative": int(row[0]), "Administrative_Duration": float(row[1]),
        "Informational": int(row[2]), "Informational_Duration": float(row[3]),
        "ProductRelated": int(row[4]), "ProductRelated_Duration": float(row[5]),
        "BounceRates": float(row[6]), "ExitRates": float(row[7]),
        "PageValues": float(row[8]), "SpecialDay": float(row[9]),
        "Month": row[10].strip(), "OperatingSystems": int(row[11]),
        "Browser": int(row[12]), "Region": int(row[13]), "TrafficType": int(row[14]),
        "VisitorType": row[15].strip(),
        "Weekend": row[16].strip().upper() == "TRUE",
        "Revenue": row[17].strip().upper() == "TRUE",
    }


def get_producer(servers):
    from kafka import KafkaProducer

    return KafkaProducer(
        bootstrap_servers=servers,
        value_serializer=lambda value: json.dumps(value).encode("utf-8"),
        request_timeout_ms=10000,
        max_block_ms=10000,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--bootstrap", default=os.environ.get("KAFKA_BOOTSTRAP", "localhost:9092"))
    ap.add_argument("--topic", default=os.environ.get("KAFKA_TOPIC", "ecommerce-sessions"))
    args = ap.parse_args()

    if not math.isfinite(args.rate) or args.rate <= 0:
        ap.error("--rate must be finite and greater than zero")
    if args.limit < 0:
        ap.error("--limit cannot be negative")

    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    raw_csv = os.path.join(base, "data", "raw", "online_shoppers_intention.csv")

    with open(raw_csv, encoding="utf-8") as f:
        rows = list(csv.reader(f))[1:]
    if not rows:
        raise ValueError(f"No session rows found in {raw_csv}")

    producer = get_producer(args.bootstrap)
    count, idx = 0, 0
    try:
        while True:
            event = build_event(rows[idx])
            idx = (idx + 1) % len(rows)
            producer.send(args.topic, value=event).get(timeout=10)

            count += 1
            if count % 20 == 0:
                print(f"[PRODUCER] Streamed {count} sessions")
            if args.limit and count >= args.limit:
                break
            time.sleep(args.rate)
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        producer.flush()
        producer.close()


if __name__ == "__main__":
    main()