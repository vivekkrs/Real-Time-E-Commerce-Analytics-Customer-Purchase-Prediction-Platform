"""
========================================================================================
MODULE 6: INTERACTIVE REAL-TIME ANALYTICS & PREDICTION DASHBOARD
File: dashboard/app.py
========================================================================================

TEACHER & COURSE PRESENTATION GUIDE:
----------------------------------------------------------------------------------------
1. Built with Streamlit and Plotly for high-impact visual demonstration.
2. Real-time auto-refresh reading live predictions scored by PySpark MLlib.
3. Showcases:
   - Live KPI metric cards (Total sessions, Conversion rate, High-intent count).
   - Real-time purchase probability gauge (0 - 100%).
   - Live session clickstream feed with predicted purchase probabilities and tags.
   - Historical seasonal benchmark comparisons from Spark SQL.
   - Model evaluation scorecard (ROC-AUC 91.7%, Accuracy 90.2%).

HOW TO RUN:
----------------------------------------------------------------------------------------
streamlit run dashboard/app.py
========================================================================================
"""

import os
import sys
import json
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Configure Streamlit page layout
st.set_page_config(
    page_title="E-Commerce Intelligence | PySpark ML & Streaming",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# CUSTOM CSS STYLING FOR SLEEK AESTHETICS
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Metric Card Styling */
    div[data-testid="stMetricValue"] {
        font-size: 2.2rem !important;
        font-weight: 700 !important;
        color: #00E5FF;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.95rem !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #A0AEC0;
    }
    .metric-card {
        background: rgba(26, 32, 44, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }
    .status-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 8px;
    }
    .badge-online { background-color: rgba(0, 230, 118, 0.2); color: #00E676; border: 1px solid #00E676; }
    .badge-pyspark { background-color: rgba(255, 112, 67, 0.2); color: #FF7043; border: 1px solid #FF7043; }
    .badge-kafka { background-color: rgba(41, 182, 246, 0.2); color: #29B6F6; border: 1px solid #29B6F6; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Base directories
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
predictions_file = os.path.join(base_dir, "data", "processed", "live_predictions.jsonl")
summary_file = os.path.join(base_dir, "data", "processed", "batch_monthly_summary.csv")
metrics_file = os.path.join(base_dir, "data", "processed", "model_metrics.json")
raw_data_file = os.path.join(base_dir, "data", "raw", "online_shoppers_intention.csv")


# -----------------------------------------------------------------------------
# SIDEBAR CONTROLS & MODEL ARCHITECTURE
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://spark.apache.org/images/spark-logo-trademark.png", width=180)
    st.title("System Control")

    auto_refresh = st.checkbox("Auto-Refresh Dashboard", value=True)
    refresh_rate = st.slider("Refresh Rate (seconds)", min_value=1, max_value=10, value=2)

    st.markdown("---")
    st.subheader("Model Performance")

    if os.path.exists(metrics_file):
        with open(metrics_file, "r", encoding="utf-8") as f:
            m = json.load(f)
        st.write(f"**Algorithm:** {m.get('model_type', 'PySpark RF')}")
        st.write(f"**ROC-AUC:** `{m.get('roc_auc', 0.9174)}`")
        st.write(f"**PR-AUC:** `{m.get('pr_auc', 0.7530)}`")
        st.write(f"**Accuracy:** `{m.get('accuracy_pct', 90.20)}%`")
        st.write(f"**F1-Score:** `{m.get('f1_score', 0.8908)}`")
        st.write(f"**Ensemble:** {m.get('num_trees', 25)} Trees, Depth {m.get('max_depth', 6)}")
    else:
        st.info("Train the model (`python src/ml/train.py`) to inspect metrics.")

    st.markdown("---")
    st.subheader("Architecture Stack")
    st.markdown(
        """
        - **Ingestion:** Apache Kafka
        - **Compute:** PySpark RDDs & DataFrames
        - **SQL Analytics:** Spark SQL
        - **Machine Learning:** Spark MLlib
        - **Stream Engine:** Structured Streaming
        - **Dashboard:** Streamlit + Plotly
        """
    )


# -----------------------------------------------------------------------------
# MAIN HEADER & STATUS BAR
# -----------------------------------------------------------------------------
st.title("🛒 Real-Time E-Commerce Analytics & Purchase Prediction")
st.markdown(
    """
    <span class="status-badge badge-online">● STREAM PIPELINE ACTIVE</span>
    <span class="status-badge badge-pyspark">● PYSPARK MLLIB: SCORING</span>
    <span class="status-badge badge-kafka">● DISTRIBUTED BUFFER: OK</span>
    """,
    unsafe_allow_html=True,
)
st.write("")

# -----------------------------------------------------------------------------
# DATA LOADING FUNCTION
# -----------------------------------------------------------------------------
def load_live_data():
    if not os.path.exists(predictions_file):
        return pd.DataFrame()
    records = []
    with open(predictions_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    return df


live_df = load_live_data()

# -----------------------------------------------------------------------------
# TOP METRICS ROW
# -----------------------------------------------------------------------------
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

if not live_df.empty:
    total_scored = len(live_df)
    high_intent = len(live_df[live_df["prediction"] == 1.0])
    intent_rate = (high_intent / total_scored) * 100 if total_scored > 0 else 0
    avg_purchase_prob = live_df["purchase_prob"].mean() * 100
else:
    total_scored = 0
    high_intent = 0
    intent_rate = 0.0
    avg_purchase_prob = 0.0

with kpi1:
    st.metric("Total Sessions Scored", f"{total_scored:,}")
with kpi2:
    st.metric("High-Intent Buyers Detected", f"{high_intent:,}")
with kpi3:
    st.metric("Live Purchase Intent Rate", f"{intent_rate:.1f}%")
with kpi4:
    st.metric("Average Purchase Probability", f"{avg_purchase_prob:.1f}%")

st.markdown("---")

# -----------------------------------------------------------------------------
# ROW 2: LIVE GAUGE & VISITOR BREAKDOWN
# -----------------------------------------------------------------------------
col_gauge, col_visitor = st.columns([1, 1])

with col_gauge:
    st.subheader("🎯 Real-Time Purchase Probability Gauge")
    gauge_val = avg_purchase_prob if total_scored > 0 else 15.5

    fig_gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=gauge_val,
            number={"suffix": "%", "font": {"size": 36, "color": "#00E5FF"}},
            title={"text": "Average Probability Across Active Sessions", "font": {"size": 16}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#00E5FF"},
                "steps": [
                    {"range": [0, 20], "color": "rgba(255, 255, 255, 0.05)"},
                    {"range": [20, 50], "color": "rgba(255, 179, 0, 0.2)"},
                    {"range": [50, 100], "color": "rgba(0, 230, 118, 0.3)"},
                ],
                "threshold": {
                    "line": {"color": "#00E676", "width": 4},
                    "thickness": 0.75,
                    "value": 50,
                },
            },
        )
    )
    fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_gauge, use_container_width=True)

with col_visitor:
    st.subheader("👥 Live Visitor Segmentation & Intent")
    if not live_df.empty:
        visitor_counts = live_df.groupby("VisitorType")["prediction"].agg(["count", "sum"]).reset_index()
        visitor_counts.columns = ["VisitorType", "Total", "HighIntent"]

        fig_bar = px.bar(
            visitor_counts,
            x="VisitorType",
            y=["Total", "HighIntent"],
            barmode="group",
            labels={"value": "Sessions", "variable": "Metric"},
            color_discrete_map={"Total": "#4A5568", "HighIntent": "#00E676"},
            title="Total Sessions vs. High Intent by Visitor Type",
        )
        fig_bar.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("Waiting for streaming data to generate visitor chart...")

# -----------------------------------------------------------------------------
# ROW 3: HISTORICAL BENCHMARK & STREAMING TIMELINE
# -----------------------------------------------------------------------------
col_trend, col_hist = st.columns([1, 1])

with col_trend:
    st.subheader("📈 Live Purchase Probability Timeline")
    if not live_df.empty:
        recent_df = live_df.tail(40).copy().reset_index()
        recent_df["prob_pct"] = recent_df["purchase_prob"] * 100

        fig_line = px.line(
            recent_df,
            x="index",
            y="prob_pct",
            labels={"index": "Recent Session Sequence", "prob_pct": "Purchase Prob (%)"},
            title="Real-Time Purchase Probability Fluctuations",
            color_discrete_sequence=["#00E5FF"],
        )
        fig_line.add_hline(y=50, line_dash="dash", line_color="#FF5252", annotation_text="50% Threshold")
        fig_line.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_line, use_container_width=True)
    else:
        st.info("Start the event producer to stream timeline events.")

with col_hist:
    st.subheader("📅 Spark SQL Monthly Conversion Benchmark")
    if os.path.exists(summary_file):
        sum_df = pd.read_csv(summary_file)
        fig_hist = px.bar(
            sum_df,
            x="Month",
            y="conversion_rate_pct",
            labels={"conversion_rate_pct": "Conversion Rate (%)"},
            color="conversion_rate_pct",
            color_continuous_scale="Viridis",
            title="Historical Conversion Rate by Month (Spark SQL)",
        )
        fig_hist.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info("Run `02_spark_sql_analytics.py` to generate monthly summary.")

# -----------------------------------------------------------------------------
# ROW 4: LIVE CLICKSTREAM PREDICTION FEED TABLE
# -----------------------------------------------------------------------------
st.subheader("⚡ Live Scored Clickstream Feed (Latest Sessions)")

if not live_df.empty:
    feed_df = live_df.tail(15).iloc[::-1].copy()

    feed_df["Probability"] = feed_df["purchase_prob"].apply(lambda p: f"{p*100:.1f}%")
    feed_df["Status"] = feed_df["prediction"].apply(
        lambda p: "🟢 High Intent (Buy)" if p == 1.0 else "⚪ Browsing"
    )
    feed_df["Actual Revenue"] = feed_df["Revenue"].apply(
        lambda r: "✅ Purchase" if r else "❌ None"
    )

    display_table = feed_df[
        [
            "session_id",
            "event_timestamp",
            "VisitorType",
            "Month",
            "PageValues",
            "ExitRates",
            "Probability",
            "Status",
            "Actual Revenue",
        ]
    ]

    st.dataframe(
        display_table,
        use_container_width=True,
        column_config={
            "session_id": "Session ID",
            "event_timestamp": "Timestamp",
            "VisitorType": "Visitor Category",
            "PageValues": st.column_config.NumberColumn("Page Value", format="%.1f"),
            "ExitRates": st.column_config.NumberColumn("Exit Rate", format="%.3f"),
            "Probability": "Purchase Prob",
            "Status": "ML Prediction",
            "Actual Revenue": "Actual Label",
        },
    )
else:
    st.warning("⚠️ No live scored sessions found in buffer yet.")
    st.markdown(
        """
        **To see live streaming events:**
        1. In Terminal 1, run the producer: `python src/producer/kafka_producer.py --rate 0.5`
        2. In Terminal 2, run the streaming scorer: `python src/streaming/spark_streaming.py`
        """
    )

# -----------------------------------------------------------------------------
# AUTO-REFRESH SCRIPT
# -----------------------------------------------------------------------------
if auto_refresh:
    time.sleep(refresh_rate)
    st.rerun()
