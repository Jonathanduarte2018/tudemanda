from __future__ import annotations

import pandas as pd
from typing import Optional


def load_time_series(csv_path: str, date_col: str = "date", value_col: str = "consumption", location_col: Optional[str] = None, location_value: Optional[str] = None) -> pd.Series:
    df = pd.read_csv(csv_path)
    if date_col not in df.columns or value_col not in df.columns:
        raise ValueError(f"CSV must contain {date_col} and {value_col} columns")

    if location_col is not None and location_value is not None:
        df = df[df[location_col] == location_value]
        if df.empty:
            raise ValueError("No rows match the requested location")

    ts = (
        df[[date_col, value_col]]
        .assign(**{date_col: pd.to_datetime(df[date_col], utc=True, errors="coerce")})
        .dropna(subset=[date_col, value_col])
        .sort_values(date_col)
        .set_index(date_col)[value_col]
    )

    # Ensure a PeriodicIndex-friendly frequency when possible
    inferred = pd.infer_freq(ts.index)
    if inferred is not None:
        ts = ts.asfreq(inferred)

    return ts
