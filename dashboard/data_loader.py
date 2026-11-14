from pathlib import Path

import pandas as pd


LIVE_COLUMNS = (
    "session_id",
    "event_timestamp",
    "VisitorType",
    "Month",
    "PageValues",
    "ExitRates",
    "prediction",
    "purchase_prob",
    "Revenue",
)


def load_live_data(output_dir, max_files=250, max_rows=10_000):
    if max_files < 1 or max_rows < 1:
        raise ValueError("max_files and max_rows must be greater than zero")

    output_path = Path(output_dir)
    if not output_path.is_dir():
        return pd.DataFrame(columns=LIVE_COLUMNS)

    files = sorted(
        (path for path in output_path.glob("*.json") if path.stat().st_size),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
    )[-max_files:]
    if not files:
        return pd.DataFrame(columns=LIVE_COLUMNS)

    frames = [pd.read_json(path, lines=True) for path in files]
    live_df = pd.concat(frames, ignore_index=True)
    missing_columns = set(LIVE_COLUMNS).difference(live_df.columns)
    if missing_columns:
        raise ValueError(
            "Prediction output is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    live_df["event_timestamp"] = pd.to_datetime(
        live_df["event_timestamp"], utc=True, errors="coerce"
    )
    if live_df["event_timestamp"].isna().any():
        raise ValueError("Prediction output contains invalid event timestamps")

    live_df["prediction"] = pd.to_numeric(live_df["prediction"], errors="raise")
    live_df["purchase_prob"] = pd.to_numeric(live_df["purchase_prob"], errors="raise")
    if not live_df["purchase_prob"].between(0, 1).all():
        raise ValueError("Purchase probabilities must be between 0 and 1")

    return (
        live_df.sort_values("event_timestamp")
        .tail(max_rows)
        .reset_index(drop=True)
    )
