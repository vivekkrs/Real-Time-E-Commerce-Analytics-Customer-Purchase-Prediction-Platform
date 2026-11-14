# E-Commerce Intelligence: End-to-End Implementation Guide & Notes

A complete, production-grade guide for building, running, and submitting the **Real-Time E-Commerce Analytics & Customer Purchase Prediction Platform** using PySpark, Kafka, Spark MLlib, Spark Structured Streaming, Docker, and Streamlit.

---

## ⚡ Master Command Cheatsheet (Quick Copy-Paste)

> **Important:** Always make sure your terminal is inside the project directory:
> ```powershell
> cd "d:\Cluster Computing\Assignment\ecommerce-intelligence"
> ```

### 🐳 Method A: One-Command Full Docker Run (Recommended)
| Task | PowerShell Command |
| :--- | :--- |
| **Launch Entire Platform** | `docker compose up --build -d` |
| **Check Container Status** | `docker compose ps` |
| **Open Dashboard UI** | Browse to `http://localhost:8501` |
| **View Live Streaming Logs** | `docker compose logs -f streaming` |
| **View Live Producer Logs** | `docker compose logs -f producer` |
| **View Dashboard Logs** | `docker compose logs -f dashboard` |
| **Stop All Containers** | `docker compose down` |

---

### 💻 Method B: Local Execution (Step-by-Step in Virtual Environment)
| Task / Script | PowerShell Command |
| :--- | :--- |
| **1. Activate Virtual Environment** | `.\cluster\Scripts\Activate.ps1` |
| **2. Run Module 1 (PySpark RDDs)** | `python src/analytics/01_rdd_analytics.py` |
| **3. Run Module 2 (Spark SQL)** | `python src/analytics/02_spark_sql_analytics.py` |
| **4. Run Module 3 (MLlib Model Train)** | `python src/ml/train.py` |
| **5. Run Tests (Pytest)** | `pytest tests/test_pipeline.py` |
| **6. Terminal 1 (Producer)** | `python src/producer/kafka_producer.py --rate 0.8` |
| **7. Terminal 2 (Spark Streaming)** | `python src/streaming/spark_streaming.py` |
| **8. Terminal 3 (Streamlit Dashboard)** | `streamlit run dashboard/app.py` |

---

## 📋 Table of Contents

1. [Project Overview & Architecture](#1-project-overview--architecture)
2. [Folder Structure](#2-folder-structure)
3. [System Prerequisites](#3-system-prerequisites)
4. [Step 1: Virtual Environment Setup](#step-1-virtual-environment-setup)
5. [Step 2: One-Command Docker Deployment](#step-2-one-command-docker-deployment)
6. [Step 3: PySpark RDD Analytics (Module 1)](#step-3-pyspark-rdd-analytics-module-1)
7. [Step 4: PySpark DataFrames & Spark SQL (Module 2)](#step-4-pyspark-dataframes--spark-sql-module-2)
8. [Step 5: PySpark MLlib Distributed Model Training (Module 3)](#step-5-pyspark-mllib-distributed-model-training-module-3)
9. [Step 6: Real-Time Kafka Event Producer (Module 4)](#step-6-real-time-kafka-event-producer-module-4)
10. [Step 7: PySpark Structured Streaming & Live Scoring (Module 5)](#step-7-pyspark-structured-streaming--live-scoring-module-5)
11. [Step 8: Interactive Streamlit Dashboard (Module 6)](#step-8-interactive-streamlit-dashboard-module-6)
12. [Step 9: Running Automated Tests](#step-9-running-automated-tests)
13. [Assignment Defense & Viva Questions](#assignment-defense--viva-questions)

---

## 1. Project Overview & Architecture

- **Objective:** Ingest real-time e-commerce clickstream sessions, compute live operational metrics using distributed processing, and predict session purchase probability (`Revenue: TRUE/FALSE`) using a pre-trained **PySpark MLlib** pipeline model.
- **Core Technology Stack:**
  - **Event Streaming:** Apache Kafka (Docker Compose)
  - **Batch & Foundations:** PySpark RDDs (map/reduceByKey), PySpark DataFrames, Spark SQL
  - **Machine Learning:** PySpark MLlib (`Pipeline`, `VectorAssembler`, `RandomForestClassifier`, `BinaryClassificationEvaluator`)
  - **Stream Processing:** Spark Structured Streaming (`readStream` from Kafka, `.transform(model)`, Parquet/JSON sink)
  - **Dashboard & UI:** Streamlit + Plotly
  - **Containerization:** Docker & Docker Compose

---

## 2. Folder Structure

```text
ecommerce-intelligence/
├── dashboard/
│   └── app.py                        # Streamlit + Plotly interactive live monitoring dashboard
├── data/
│   ├── processed/
│   │   ├── batch_monthly_summary.csv # Spark SQL aggregated monthly metrics
│   │   ├── live_predictions.jsonl    # Real-time scored sessions with purchase probabilities
│   │   └── model_metrics.json        # Test evaluation report (ROC-AUC 91.74%, Acc 90.20%)
│   └── raw/
│       └── online_shoppers_intention.csv # Source UCI benchmark dataset
├── docs/
│   └── project_report.md             # Complete academic submission report (Markdown)
├── models/
│   └── spark_purchase_model/         # Serialized PySpark MLlib PipelineModel
├── notebooks/
│   └── exploratory_analysis.ipynb    # Jupyter Notebook for interactive EDA & Spark SQL
├── src/
│   ├── analytics/
│   │   ├── 01_rdd_analytics.py       # Module 1: PySpark RDDs (map/reduceByKey/filter)
│   │   └── 02_spark_sql_analytics.py # Module 2: DataFrames, SQL views & Window Functions
│   ├── ml/
│   │   └── train.py                  # Module 3: Distributed PySpark MLlib Pipeline
│   ├── producer/
│   │   └── kafka_producer.py         # Module 4: Live Event Producer & Kafka publisher
│   └── streaming/
│       └── spark_streaming.py        # Module 5: Structured Streaming & Live ML scoring
├── tests/
│   └── test_pipeline.py              # Automated pipeline sanity tests
├── docker-compose.yml                # Docker Compose (Kafka, Producer, Streaming, Dashboard)
├── Dockerfile                        # Multi-stage image with Python 3.11 & OpenJDK 17
├── .dockerignore                     # Docker build exclusion rules
├── requirements.txt                  # Python dependencies
└── Notes.md                          # This comprehensive walkthrough guide
```

---

## 3. System Prerequisites

Before running the commands, verify that you have:

1. **Python 3.10+ installed:**
   ```powershell
   python --version
   ```
2. **Java 8, 11, or 17 (Required by Apache Spark):**
   ```powershell
   java -version
   ```
3. **Docker Desktop installed and running:**
   ```powershell
   docker --version
   docker ps
   ```

---

## Step 1: Virtual Environment Setup

Open PowerShell in the `ecommerce-intelligence` root folder:

```powershell
# Navigate into the project folder
cd "d:\Cluster Computing\Assignment\ecommerce-intelligence"

# Create a dedicated virtual environment named cluster
python -m venv cluster

# Activate the virtual environment on Windows
.\cluster\Scripts\Activate.ps1
```

> **Note for Execution Policy Error:**
> If PowerShell blocks script activation with an `Execution_Policies` error, run:
> ```powershell
> Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
> .\cluster\Scripts\Activate.ps1
> ```

Upgrade pip and install the required dependencies:
```powershell
python -m pip install --upgrade pip setuptools
pip install -r requirements.txt
```

---

## Step 2: One-Command Docker Deployment

You can launch the **entire end-to-end platform with a single command** using Docker Compose (Kafka, Producer, PySpark Streaming, and Streamlit Dashboard):

```powershell
# Build and run all 4 containers in the background
docker compose up --build -d

# Check live status of all services
docker compose ps
```

* **Dashboard Web UI:** Open [http://localhost:8501](http://localhost:8501)
* **View Real-Time Logs:**
  ```powershell
  # View all service logs cleanly
  docker compose logs -f

  # Or view a specific container's logs:
  docker compose logs -f streaming
  docker compose logs -f producer
  docker compose logs -f dashboard
  ```
* **Stop Everything cleanly:**
  ```powershell
  docker compose down
  ```

---

## Step 3: PySpark RDD Analytics (Module 1)

- **Script:** `src/analytics/01_rdd_analytics.py`
- **Academic Goal:** Demonstrate low-level distributed primitives (`map`, `filter`, `reduceByKey`, `groupByKey`, `mapValues`) without relying on the Catalyst SQL optimizer.
- **Key Tasks:**
  1. Load raw CSV into an RDD via `sc.textFile`.
  2. Parse CSV rows and remove the header lazily.
  3. Compute average `PageValues` and average `ExitRates` by `VisitorType` using `reduceByKey`.
  4. Filter high-intent sessions (`PageValues > 50`) and count revenue conversions.

**Run Command:**
```powershell
python src/analytics/01_rdd_analytics.py
```

---

## Step 4: PySpark DataFrames & Spark SQL (Module 2)

- **Script:** `src/analytics/02_spark_sql_analytics.py`
- **Academic Goal:** Leverage Spark DataFrames, Catalyst query optimization, schema validation, and Spark SQL temporary views.
- **Key Tasks:**
  1. Read `data/raw/online_shoppers_intention.csv` with inferSchema / explicit `StructType`.
  2. Register temporary SQL table: `df.createOrReplaceTempView("sessions")`.
  3. Run analytical queries:
     - Conversion rate by `Month` and `TrafficType`.
     - Bounce rate vs. purchase correlation.
     - Window function: Rank traffic sources by conversion efficiency (`DENSE_RANK`).
  4. Save summary analytics to `data/processed/batch_monthly_summary.csv`.

**Run Command:**
```powershell
python src/analytics/02_spark_sql_analytics.py
```

---

## Step 5: PySpark MLlib Distributed Model Training (Module 3)

- **Script:** `src/ml/train.py`
- **Important Teacher Instruction:** Strictly use **PySpark MLlib** (`pyspark.ml`), not Python `scikit-learn`.
- **Distributed Pipeline Components:**
  1. `StringIndexer`: Indexes `Month` and `VisitorType`.
  2. `OneHotEncoder`: Converts indices into distributed sparse vectors.
  3. `VectorAssembler`: Combines continuous features (`PageValues`, `BounceRates`, `ExitRates`, durations) + categorical vectors into a single feature vector `features`.
  4. `StandardScaler`: Normalizes feature vectors across worker partitions.
  5. Estimator: `RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=25, maxDepth=6)`.
  6. Evaluator: `BinaryClassificationEvaluator(metricName="areaUnderROC")`.
  7. Export: Persist the trained `PipelineModel` to `models/spark_purchase_model/`.

**Run Command:**
```powershell
python src/ml/train.py
```

---

## Step 6: Real-Time Kafka Event Producer (Module 4)

- **Script:** `src/producer/kafka_producer.py`
- **Goal:** Simulate real-time e-commerce user browsing activity by reading the raw CSV dataset and publishing session events as JSON to the Kafka topic `ecommerce-sessions`.
- **Configuration:**
  - Topic: `ecommerce-sessions`
  - Speed: 1 event every 0.8s (configurable with `--rate 0.5`)
  - Adds dynamic runtime metadata: `session_id`, `event_timestamp`.

**Run Command (in a separate terminal):**
```powershell
python src/producer/kafka_producer.py --rate 0.8
```

---

## Step 7: PySpark Structured Streaming & Live Scoring (Module 5)

- **Script:** `src/streaming/spark_streaming.py`
- **Goal:** Ingest events continuously, load the pre-trained `models/spark_purchase_model`, run real-time inference using native `vector_to_array`, and write predictions to Parquet/JSON storage.
- **Features:**
  - Uses native JVM `vector_to_array` to eliminate socket timeouts.
  - Invokes `pipeline_model.transform(stream_df)` natively on the streaming DataFrame.
  - Calculates live purchase probability and intent prediction (`1` or `0`).
  - Writes streaming sink to `data/processed/live_predictions.jsonl`.

**Run Command (in another terminal):**
```powershell
python src/streaming/spark_streaming.py
```

---

## Step 8: Interactive Streamlit Dashboard (Module 6)

- **Script:** `dashboard/app.py`
- **Goal:** Real-time visual monitoring interface for stakeholders.
- **Features:**
  - **Live KPI Counters:** Total Sessions, Current Conversion Rate, High-Intent Sessions.
  - **Interactive Gauges & Trend Charts:** Real-time purchase probability gauge, exit rate distribution, and monthly conversion comparisons.
  - **Live Stream Feed:** Real-time streaming table showing latest sessions with their predicted purchase probability and status tags.

**Run Command (in a third terminal):**
```powershell
streamlit run dashboard/app.py
```

Open your browser at `http://localhost:8501`.

---

## Step 9: Running Automated Tests

Run the test suite to verify the dataset, trained models, metrics, and batch summary:

```powershell
pytest tests/test_pipeline.py -v
```

---

## Assignment Defense & Viva Questions

| Question | Recommended Answer |
| :--- | :--- |
| **Why use PySpark RDDs if DataFrames are faster?** | RDDs provide low-level functional control (`map`, `reduceByKey`, `partitionBy`) and demonstrate understanding of distributed data structures, lineage graphs (DAG), and lazy evaluation without automatic query optimization. |
| **Why is PySpark MLlib required instead of Scikit-Learn?** | Scikit-Learn collects all data into single-node Python memory (driver bottleneck). PySpark MLlib distributes matrix calculations and tree splits across all partitions and worker nodes, and exports a `PipelineModel` that can directly score live Spark Structured Streaming DataFrames via `.transform()`. |
| **What is the role of Kafka in this architecture?** | Kafka acts as a distributed, fault-tolerant message buffer. It decouples high-throughput clickstream event ingestion from the Spark Structured Streaming consumer, preventing data loss during traffic spikes. |
| **How does Spark Structured Streaming handle fault tolerance?** | Structured Streaming combines Kafka offset tracking with **checkpointing** and **write-ahead logs** (WAL). If a worker or driver fails, Spark restarts from the exact recorded offset and guarantees end-to-end exactly-once processing when writing to Parquet. |
| **Why store data in Apache Parquet format?** | Parquet is a columnar storage format with snappy compression and dictionary encoding. It allows projection pushdown and column pruning, drastically reducing disk I/O and query time for downstream analytics. |
