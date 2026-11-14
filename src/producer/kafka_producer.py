import os, sys, time, json, uuid, csv, argparse
from datetime import datetime, timezone


def get_producer(servers):
    try:
        from kafka import KafkaProducer
        p = KafkaProducer(bootstrap_servers=servers,
                          value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                          request_timeout_ms=3000, max_block_ms=3000)
        p.bootstrap_connected()
        print(f"[KAFKA] Connected: {servers}")
        return p
    except Exception:
        print("[PRODUCER] Kafka offline, using local buffer file only")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--bootstrap", default=os.environ.get("KAFKA_BOOTSTRAP", "localhost:9092"))
    ap.add_argument("--topic", default="ecommerce-sessions")
    args = ap.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    raw_csv = os.path.join(base, "data", "raw", "online_shoppers_intention.csv")
    buffer_file = os.path.join(base, "data", "processed", "live_stream_buffer.jsonl")
    os.makedirs(os.path.dirname(buffer_file), exist_ok=True)

    with open(raw_csv, encoding="utf-8") as f:
        rows = list(csv.reader(f))[1:]          # header skip

    producer = get_producer(args.bootstrap)
    count, idx = 0, 0
    try:
        while True:
            r = rows[idx]; idx = (idx + 1) % len(rows)
            event = {
                "session_id": f"SES-{uuid.uuid4().hex[:8].upper()}",
                "event_timestamp": datetime.now(timezone.utc).isoformat(),
                "Administrative": int(r[0]), "Administrative_Duration": float(r[1]),
                "Informational": int(r[2]), "Informational_Duration": float(r[3]),
                "ProductRelated": int(r[4]), "ProductRelated_Duration": float(r[5]),
                "BounceRates": float(r[6]), "ExitRates": float(r[7]),
                "PageValues": float(r[8]), "SpecialDay": float(r[9]),
                "Month": r[10].strip(), "OperatingSystems": int(r[11]),
                "Browser": int(r[12]), "Region": int(r[13]), "TrafficType": int(r[14]),
                "VisitorType": r[15].strip(),
                "Weekend": r[16].strip().upper() == "TRUE",
                "Revenue": r[17].strip().upper() == "TRUE",
            }
            if producer:
                try: producer.send(args.topic, value=event)
                except Exception: pass
            with open(buffer_file, "a", encoding="utf-8") as bf:
                bf.write(json.dumps(event) + "\n")

            count += 1
            if count % 20 == 0:
                print(f"[PRODUCER] Streamed {count} sessions")
            if args.limit and count >= args.limit:
                break
            time.sleep(args.rate)
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        if producer:
            producer.flush(); producer.close()


if __name__ == "__main__":
    main()