"""
Data preparation layer for the Krannert dashboard.

Provides reusable functions to ingest CSVs, standardize schema, and compute
aggregate tables that power Streamlit visuals.
"""
from __future__ import annotations

from pathlib import Path
from typing import IO, Iterable

import numpy as np
import pandas as pd
import streamlit as st

from . import utils

# ---------------------------------------------------------------------------
# Event type mappings
# ---------------------------------------------------------------------------
SERIES_LOOKUP = {
    "gala": "Gala",
    "chamber": "Chamber Series",
    "dance": "Dance Series",
    "family": "Family Series",
    "lecture": "Lecture Series",
    "opera": "Opera Series",
    "symphony": "Symphony Series",
    "ballet": "Ballet Series",
}

KEYWORD_TYPE_MAP = [
    ("symphony", "Symphony Orchestra"),
    ("orchestra", "Symphony Orchestra"),
    ("choir", "Choir / Vocal"),
    ("chorus", "Choir / Vocal"),
    ("glee", "Choir / Vocal"),
    ("opera", "Opera"),
    ("ballet", "Ballet"),
    ("dance", "Dance"),
    ("jazz", "Jazz"),
    ("recital", "Recital"),
    ("talk", "Talk / Lecture"),
    ("lecture", "Talk / Lecture"),
    ("festival", "Festival"),
    ("film", "Film / Screening"),
]

REQUIRED_COLUMNS = {"sale_date", "event_date", "event_name", "qty_sold"}
OPTIONAL_NUMERIC = ("qty_sold", "amount")

LEAD_BUCKETS: tuple[tuple[int, int, str], ...] = (
    (0, 7, "0-7"),
    (8, 14, "8-14"),
    (15, 30, "15-30"),
    (31, 60, "31-60"),
    (61, 120, "61-120"),
)

MAX_LEAD_DAYS = 120
D_BIN_SIZE = 3  # 3-day bins for cohort aggregation
MIN_COHORT_SIZE = 20


# ---------------------------------------------------------------------------
# Header / column utilities
# ---------------------------------------------------------------------------
def _clean_headers(columns: Iterable[str]) -> list[str]:
    """Lowercase, strip, and snake_case column names."""
    cleaned = []
    for col in columns:
        text = str(col).strip().lower().replace(" ", "_")
        cleaned.append(text)
    return cleaned


def _enforce_required(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for col in OPTIONAL_NUMERIC:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _coerce_datetimes(df: pd.DataFrame) -> pd.DataFrame:
    for col in ("sale_date", "event_date"):
        df[col] = pd.to_datetime(df[col], errors="coerce", utc=False)
    return df


def _ensure_event_category(df: pd.DataFrame) -> pd.DataFrame:
    if "event_category" not in df.columns:
        df["event_category"] = df["event_name"].apply(lambda name: utils.categorize_event(None, name))
    else:
        df["event_category"] = df["event_category"].apply(utils.categorize_event)
    return df


def _derive_event_type(row: pd.Series) -> str:
    """Map event_part codes to human names, then keyword-infer for leftovers."""
    part = row.get("event_part")
    if isinstance(part, str) and part.strip():
        key = part.strip().lower()
        if key in SERIES_LOOKUP:
            return SERIES_LOOKUP[key]
        # Keyword match against part string
        for keyword, label in KEYWORD_TYPE_MAP:
            if keyword in key:
                return label
        return part.strip().title()

    name = str(row.get("event_name", "") or "")
    lowered = name.lower()
    for keyword, label in KEYWORD_TYPE_MAP:
        if keyword in lowered:
            return label

    category = row.get("event_category")
    if isinstance(category, str) and category.strip():
        return category.strip()
    return "Other"


def _append_event_type(df: pd.DataFrame) -> pd.DataFrame:
    df["event_type"] = df.apply(_derive_event_type, axis=1)
    return df


def _compose_event_label(df: pd.DataFrame) -> pd.Series:
    label = df["event_name"].fillna("Unknown").astype(str).str.strip()
    if "event_part" in df.columns:
        part = df["event_part"].fillna("").astype(str).str.strip()
        label = np.where(part == "", label, label + " – " + part)
    return label


def _compose_event_instance(df: pd.DataFrame, label_col: str = "event_label") -> pd.Series:
    date_part = df["event_date"].dt.strftime("%Y-%m-%d")
    return (
        df[label_col].str.lower().str.strip().fillna("unknown")
        + "|"
        + date_part.fillna("unknown")
    )


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_csv(path_or_buffer: str | Path | IO[str]) -> pd.DataFrame:
    """
    Read a CSV, normalize schema, and ensure required columns exist.
    """
    df = pd.read_csv(path_or_buffer)
    df.columns = _clean_headers(df.columns)
    _enforce_required(df)
    df = _coerce_numeric(df)
    df = _coerce_datetimes(df)

    df = df.dropna(subset=["sale_date", "event_date", "event_name", "qty_sold"]).copy()
    df["event_name"] = df["event_name"].astype(str).str.strip()
    df["qty_sold"] = df["qty_sold"].fillna(0)

    df = _ensure_event_category(df)
    df = _append_event_type(df)
    df["event_label"] = _compose_event_label(df)
    df["event_instance"] = _compose_event_instance(df, "event_label")
    df["weekday"] = df["event_date"].dt.day_name()
    if "venue" in df.columns:
        df["venue"] = df["venue"].fillna("Unknown").astype(str).str.strip().replace({"": "Unknown"})
    else:
        df["venue"] = "Unknown"
    return df


# ---------------------------------------------------------------------------
# Synthetic data for development
# ---------------------------------------------------------------------------
def _generate_qty_splits(total: int, count: int, rng: np.random.Generator) -> np.ndarray:
    weights = rng.random(count)
    weights = weights / weights.sum()
    qtys = np.floor(weights * total).astype(int)
    remainder = total - qtys.sum()
    if remainder > 0:
        qtys[:remainder] += 1
    return qtys


def make_fake_data(n_events: int = 50, seed: int | None = 42) -> pd.DataFrame:
    """
    Fabricate a synthetic dataset for local development previews.
    """
    rng = np.random.default_rng(seed)
    pre_dates = pd.date_range("2018-01-01", "2019-12-31")
    post_dates = pd.date_range("2022-01-01", "2024-12-31")

    pre_count = max(1, n_events // 2)
    post_count = max(1, n_events - pre_count)
    event_dates = list(rng.choice(pre_dates, size=pre_count))
    event_dates.extend(rng.choice(post_dates, size=post_count))
    rng.shuffle(event_dates)

    categories = ["Theatre", "Concerts", "Talks", "Festivals", "Film"]
    channels = ["Web", "Phone", "Box Office", "Group Sales"]
    customers = ["Subscriber", "Single Ticket", "Student", "Group"]
    parts = ["Matinee", "Evening", "Opening Night", ""]

    records: list[dict] = []
    for idx, raw_date in enumerate(event_dates[:n_events]):
        event_date = pd.Timestamp(raw_date)
        event_name = f"Sample Event {idx + 1}"
        event_category = rng.choice(categories)
        price = rng.integers(35, 180)
        season = f"{event_date.year}-{str(event_date.year + 1)[-2:]}"
        sale_points = int(rng.integers(10, 30))
        total_qty = int(rng.integers(200, 2500))
        lead_days = np.sort(rng.integers(0, 150, size=sale_points))
        qtys = _generate_qty_splits(total_qty, sale_points, rng)
        event_part = rng.choice(parts)
        event_part = event_part or None

        for lead, qty in zip(lead_days, qtys):
            sale_date = event_date - pd.to_timedelta(int(lead), unit="D")
            records.append(
                {
                    "sale_date": sale_date,
                    "event_date": event_date,
                    "event_name": event_name,
                    "event_part": event_part,
                    "qty_sold": int(max(qty, 1)),
                    "amount": float(max(qty, 1) * price),
                    "channel": rng.choice(channels),
                    "customer_type": rng.choice(customers),
                    "season": season,
                    "event_category": event_category,
                }
            )

    df = pd.DataFrame(records)
    df["sale_date"] = pd.to_datetime(df["sale_date"])
    df["event_date"] = pd.to_datetime(df["event_date"])
    df["qty_sold"] = df["qty_sold"].astype(int)
    df["amount"] = df["amount"].astype(float)
    df = _ensure_event_category(df)
    df = _append_event_type(df)
    df["event_label"] = _compose_event_label(df)
    df["event_instance"] = _compose_event_instance(df, "event_label")
    df["weekday"] = df["event_date"].dt.day_name()
    df["venue"] = "Generated Venue"
    return df.sort_values(["event_date", "sale_date"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Lead time / pacing calculations
# ---------------------------------------------------------------------------
def build_lead_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach lead-day calculations for every transaction row.
    days_out = (event_date - sale_date).days
    """
    work = df.copy()
    work["days_out"] = (work["event_date"].dt.normalize() - work["sale_date"].dt.normalize()).dt.days
    return work


def compute_event_pacing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-event cumulative sell-in with 3-day bins.
    
    Returns DataFrame with: event_instance, event_type, weekday, venue, d_bin, cum_pct, event_date
    """
    if "days_out" not in df.columns:
        df = build_lead_time(df)
    
    s = df[(df["days_out"] >= 0) & (df["days_out"] <= MAX_LEAD_DAYS)].copy()
    if s.empty:
        return pd.DataFrame(columns=["event_instance", "event_type", "weekday", "venue", "d_bin", "cum_pct", "event_date"])
    
    # 3-day bins
    s["d_bin"] = (s["days_out"] // D_BIN_SIZE) * D_BIN_SIZE
    
    # Sort by event_instance, days_out DESCENDING (120 → 0)
    s = s.sort_values(["event_instance", "days_out"], ascending=[True, False])
    
    # Cumsum per event (from earliest sale to latest = days 120 → 0)
    s["cum_qty"] = s.groupby("event_instance", sort=False)["qty_sold"].cumsum()
    
    # Final total per event
    finals = s.groupby("event_instance")["cum_qty"].max()
    s["cum_pct"] = np.where(
        s["event_instance"].map(finals) > 0,
        (s["cum_qty"] / s["event_instance"].map(finals)) * 100,
        np.nan
    )
    
    # Keep one row per (event_instance, d_bin) - the latest cum_pct at that bin
    aggregated = (
        s.groupby(["event_instance", "d_bin"])
        .agg(
            event_type=("event_type", "first"),
            weekday=("weekday", "first"),
            venue=("venue", "first"),
            event_date=("event_date", "first"),
            cum_pct=("cum_pct", "max"),  # max cum_pct within that bin
        )
        .reset_index()
    )
    
    return aggregated


def build_cohort_library(pacing_df: pd.DataFrame, today: pd.Timestamp) -> dict[str, pd.DataFrame]:
    """
    Build cohort statistics library for median lookups with fallback levels.
    Uses historical data only (events before today).
    """
    history = pacing_df[pacing_df["event_date"] < today].copy()
    
    def compute_stats(data: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
        if data.empty:
            cols = group_cols + ["d_bin", "median_cum_pct", "p25_cum_pct", "p75_cum_pct", "sample_size"]
            return pd.DataFrame(columns=cols)
        
        stats = (
            data.groupby(group_cols + ["d_bin"])
            .agg(
                median_cum_pct=("cum_pct", "median"),
                p25_cum_pct=("cum_pct", lambda x: np.nanpercentile(x.dropna(), 25) if len(x.dropna()) else np.nan),
                p75_cum_pct=("cum_pct", lambda x: np.nanpercentile(x.dropna(), 75) if len(x.dropna()) else np.nan),
                sample_size=("event_instance", "nunique"),
            )
            .reset_index()
        )
        return stats
    
    library = {
        "type_day_venue": compute_stats(history, ["event_type", "weekday", "venue"]),
        "type_day": compute_stats(history, ["event_type", "weekday"]),
        "type_only": compute_stats(history, ["event_type"]),
        "global": compute_stats(history, []),
    }
    return library


def build_global_pacing_curve(pacing_df: pd.DataFrame, today: pd.Timestamp) -> pd.DataFrame:
    """
    Build global pacing curve (median + IQR) by d_bin for the booking window chart.
    """
    history = pacing_df[pacing_df["event_date"] < today].copy()
    if history.empty:
        return pd.DataFrame(columns=["d_bin", "median_cum_pct", "p25_cum_pct", "p75_cum_pct"])
    
    stats = (
        history.groupby("d_bin")
        .agg(
            median_cum_pct=("cum_pct", "median"),
            p25_cum_pct=("cum_pct", lambda x: np.nanpercentile(x.dropna(), 25) if len(x.dropna()) else np.nan),
            p75_cum_pct=("cum_pct", lambda x: np.nanpercentile(x.dropna(), 75) if len(x.dropna()) else np.nan),
        )
        .reset_index()
    )
    
    # Fill in all bins 0 to 120
    all_bins = pd.DataFrame({"d_bin": range(0, MAX_LEAD_DAYS + 1, D_BIN_SIZE)})
    stats = all_bins.merge(stats, on="d_bin", how="left")
    
    # Forward-fill and enforce monotonicity (curves should rise toward 100% at d_bin=0)
    for col in ("p25_cum_pct", "median_cum_pct", "p75_cum_pct"):
        stats = stats.sort_values("d_bin", ascending=False)  # 120 → 0
        stats[col] = stats[col].ffill().fillna(0)
        # Cummax ensures monotonic increase as we approach event day
        stats[col] = stats[col].cummax().clip(0, 100)
    
    # Ensure p25 ≤ median ≤ p75
    stats["p25_cum_pct"] = stats["p25_cum_pct"].clip(upper=stats["median_cum_pct"])
    stats["p75_cum_pct"] = stats["p75_cum_pct"].clip(lower=stats["median_cum_pct"])
    
    return stats.sort_values("d_bin", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Share / avg tickets utilities
# ---------------------------------------------------------------------------
def category_share(df: pd.DataFrame, group_col: str = "event_type", label: str = "event_type") -> pd.DataFrame:
    """Normalize category totals so that shares sum to ~1.0."""
    if df.empty:
        return pd.DataFrame(columns=[label, "share_ratio"])
    totals = (
        df.groupby(group_col, dropna=False)["qty_sold"]
        .sum()
        .reset_index()
        .sort_values("qty_sold", ascending=False)
    )
    grand_total = totals["qty_sold"].sum()
    if grand_total == 0:
        totals["share_ratio"] = 0.0
    else:
        totals["share_ratio"] = totals["qty_sold"] / grand_total
    return totals[[group_col, "share_ratio"]].rename(columns={group_col: label})


def average_tickets_per_event(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Average tickets per event for a given dataframe slice."""
    if df.empty:
        return pd.DataFrame(columns=["event_type", "avg_tickets", "window"])
    grouped = (
        df.groupby("event_type")
        .agg(total_qty=("qty_sold", "sum"), event_count=("event_instance", "nunique"))
        .reset_index()
    )
    grouped = grouped[grouped["event_count"] > 0]
    grouped["avg_tickets"] = grouped["total_qty"] / grouped["event_count"]
    grouped["window"] = label
    return grouped[["event_type", "avg_tickets", "window"]]


def events_common(pre_df: pd.DataFrame, post_df: pd.DataFrame, top_n: int = 12) -> pd.DataFrame:
    """Events that appear in both windows, ranked by combined volume."""
    if pre_df.empty or post_df.empty:
        return pd.DataFrame(columns=["event_name", "qty_pre", "qty_post"])

    pre_totals = pre_df.groupby("event_name")["qty_sold"].sum()
    post_totals = post_df.groupby("event_name")["qty_sold"].sum()
    common_events = pre_totals.index.intersection(post_totals.index)
    if common_events.empty:
        return pd.DataFrame(columns=["event_name", "qty_pre", "qty_post"])

    merged = pd.DataFrame(
        {
            "event_name": common_events,
            "qty_pre": pre_totals.loc[common_events].values,
            "qty_post": post_totals.loc[common_events].values,
        }
    )
    merged["qty_combined"] = merged["qty_pre"] + merged["qty_post"]
    merged = merged.sort_values("qty_combined", ascending=False).head(top_n)
    return merged.drop(columns="qty_combined").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Heatmap utilities
# ---------------------------------------------------------------------------
def _lead_bucket(value: float | int | None) -> str:
    if value is None or np.isnan(value):
        return "unknown"
    for low, high, label in LEAD_BUCKETS:
        if low <= value <= high:
            return label
    return ">120" if value > LEAD_BUCKETS[-1][1] else "<0"


def _weekday_categorical(series: pd.Series) -> pd.Series:
    ordered = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    return pd.Categorical(series, categories=ordered, ordered=True)


def build_sales_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build weekday × lead bucket heatmap showing % of weekly sales.
    """
    if "days_out" not in df.columns:
        return pd.DataFrame(columns=["weekday", "lead_bucket", "share"])
    
    window = df[df["days_out"].between(0, LEAD_BUCKETS[-1][1])].copy()
    if window.empty:
        return pd.DataFrame(columns=["weekday", "lead_bucket", "share"])
    
    window["weekday"] = window["sale_date"].dt.day_name()
    window["lead_bucket"] = window["days_out"].apply(_lead_bucket)
    
    out = (
        window.groupby(["weekday", "lead_bucket"], as_index=False)["qty_sold"]
        .sum()
    )
    
    # % of weekday sales per bucket
    totals = out.groupby("weekday")["qty_sold"].transform("sum")
    out["share"] = np.where(totals > 0, out["qty_sold"] / totals, 0)
    out["weekday"] = _weekday_categorical(out["weekday"])
    out = out.sort_values(["weekday", "lead_bucket"]).reset_index(drop=True)
    return out[["weekday", "lead_bucket", "share"]]


# ---------------------------------------------------------------------------
# Main aggregation
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def derive_core(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Produce a dictionary of cleaned aggregates ready for plotting.
    """
    base = build_lead_time(df)
    today = pd.Timestamp.today().normalize()
    
    # Pre/post windows
    pre_df = base[base["sale_date"] < utils.PRE_WINDOW_END].copy()
    post_df = base[base["sale_date"] >= utils.POST_WINDOW_START].copy()
    
    # Event pacing curves
    pacing = compute_event_pacing(base)
    cohort_library = build_cohort_library(pacing, today)
    global_curve = build_global_pacing_curve(pacing, today)
    
    # Avg tickets
    avg_pre = average_tickets_per_event(pre_df, "Pre (sale < Mar 1 2020)")
    avg_post = average_tickets_per_event(post_df, "Post (sale ≥ Jul 1 2021)")
    
    outputs = {
        "base": base,
        "event_pacing": pacing,
        "cohort_library": cohort_library,
        "global_pacing_curve": global_curve,
        "avg_tickets": pd.concat([avg_pre, avg_post], ignore_index=True),
        "category_share_pre": category_share(pre_df, group_col="event_type", label="event_type"),
        "category_share_post": category_share(post_df, group_col="event_type", label="event_type"),
        "events_pre_post": events_common(pre_df, post_df),
        "dows_heatmap": build_sales_heatmap(base),
    }
    return outputs
