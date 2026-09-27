"""Statistical transformations on indicator DataFrames returned by db_manager."""

from __future__ import annotations

from functools import reduce

import numpy as np
import pandas as pd


def add_growth_columns(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """Add annualised growth (%) and rolling standard deviation to one series.

    Growth is annualised across reporting gaps, so a 2015 -> 2020 step is
    comparable with a single-year step.
    """
    frame = df.dropna(subset=["value"]).sort_values("year").reset_index(drop=True)
    ratio = frame["value"] / frame["value"].shift()
    gap = frame["year"].astype(float).diff()
    growth = (ratio ** (1 / gap) - 1) * 100
    frame["growth_pct"] = growth.replace([np.inf, -np.inf], np.nan)
    frame["rolling_std"] = frame["value"].rolling(window, min_periods=window).std()
    return frame


def _year_of_max(frame: pd.DataFrame, column: str) -> int | None:
    values = frame[column].dropna()
    if values.empty:
        return None
    return int(frame.loc[values.idxmax(), "year"])


def compute_summary_stats(df: pd.DataFrame, window: int = 3) -> dict:
    """Descriptive statistics, total growth, peak growth year and most volatile year."""
    frame = add_growth_columns(df, window)
    values = frame["value"]
    first = values.iloc[0] if len(values) else np.nan
    last = values.iloc[-1] if len(values) else np.nan
    return {
        "observations": int(values.count()),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "median": float(values.median()),
        "min": float(values.min()),
        "max": float(values.max()),
        "total_growth_pct": float((last / first - 1) * 100) if first else np.nan,
        "peak_growth_year": _year_of_max(frame, "growth_pct"),
        "peak_growth_pct": float(frame["growth_pct"].max()),
        "most_volatile_year": _year_of_max(frame, "rolling_std"),
    }


def build_wide_frame(df_list: list[pd.DataFrame]) -> pd.DataFrame:
    """Join single-indicator series for one country into a year x indicator table."""
    columns = []
    for df in df_list:
        clean = df.dropna(subset=["value"])
        if clean.empty:
            continue
        code = clean["indicator_code"].iloc[0]
        columns.append(clean.groupby("year")["value"].mean().rename(code).reset_index())
    if not columns:
        return pd.DataFrame()
    wide = reduce(lambda left, right: pd.merge(left, right, on="year", how="outer"), columns)
    return wide.sort_values("year").set_index("year")


def build_correlation_matrix(
    df_list: list[pd.DataFrame],
    method: str = "pearson",
    min_periods: int = 3,
) -> pd.DataFrame:
    """Pairwise correlation between indicators, using only years both report."""
    return build_wide_frame(df_list).corr(method=method, min_periods=min_periods)


def shared_years(df: pd.DataFrame, group: str = "country_name") -> list[int]:
    """Years in which every group in the DataFrame has a value."""
    clean = df.dropna(subset=["value"])
    counts = clean.groupby("year")[group].nunique()
    return [int(year) for year in counts[counts == clean[group].nunique()].index]


def calculate_normalized_growth(df: pd.DataFrame, base_year: int) -> pd.DataFrame:
    """Express each country's values as % change from its own base-year value.

    Countries without a value in the base year are dropped.
    """
    clean = df.dropna(subset=["value"])
    base = clean.loc[clean["year"] == base_year, ["country_name", "value"]]
    merged = clean.merge(base.rename(columns={"value": "base_value"}), on="country_name")
    growth = (merged["value"] / merged["base_value"] - 1) * 100
    merged["growth_pct"] = growth.replace([np.inf, -np.inf], np.nan)
    return (
        merged.drop(columns="base_value")
        .sort_values(["country_name", "year"])
        .reset_index(drop=True)
    )
