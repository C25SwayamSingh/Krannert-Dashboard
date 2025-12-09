"""
Pacing module for the Krannert dashboard.

Fixes the 100% issue by comparing:
  - Cum % = tickets_so_far / median_final_of_cohort (not event's own final)
  - Median @D = historical median cum% at the same days-out bin from cohort

This gives meaningful pacing comparison for upcoming events.
"""
from __future__ import annotations

from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

# ====== TUNABLES ======
D_MAX = 120                        # only pace up to 120 days out
BIN = 3                            # use 3-day lead bins
AHEAD_PP = 5.0                     # ±5 percentage-points bands
MIN_COHORT = 20                    # min past events needed before fallback
HIDE_SOLDOUT_AT = 98.0             # hide rows that are effectively sold out

# most-specific → least-specific fallback tiers
DEFAULT_COHORT_TIERS: List[List[str]] = [
    ["event_type", "weekday", "venue"],  # best
    ["event_type", "weekday"],
    ["event_type"],
    [],  # global fallback
]

TIER_LABELS = {
    "event_type|weekday|venue": "Type+Weekday+Venue",
    "event_type|weekday": "Type+Weekday",
    "event_type": "Type only",
    "all": "All events (global)",
}


def _format_cohort_label(tier: str, n: int, row: pd.Series) -> str:
    """
    Format a human-readable cohort label with context.
    Examples:
      - Type+Weekday+Venue (n=45): Theatre · Saturday · Main Hall
      - Type+Weekday (n=32): Theatre · Saturday
      - Type only (n=120): Theatre
      - All events (global) (n=500)
    """
    base_label = TIER_LABELS.get(tier, tier)
    
    if tier == "event_type|weekday|venue":
        event_type = row.get("event_type", "Unknown")
        weekday = row.get("weekday", "Unknown")
        venue = row.get("venue", "Unknown")
        return f"{base_label} (n={n}): {event_type} · {weekday} · {venue}"
    elif tier == "event_type|weekday":
        event_type = row.get("event_type", "Unknown")
        weekday = row.get("weekday", "Unknown")
        return f"{base_label} (n={n}): {event_type} · {weekday}"
    elif tier == "event_type":
        event_type = row.get("event_type", "Unknown")
        return f"{base_label} (n={n}): {event_type}"
    else:
        return f"{base_label} (n={n})"


def _prep(df: pd.DataFrame, today: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """Derive days_out, d_bin, weekday. Keep only rows we can use."""
    df = df.copy()
    df["sale_date"] = pd.to_datetime(df["sale_date"])
    df["event_date"] = pd.to_datetime(df["event_date"])

    if "weekday" not in df.columns:
        df["weekday"] = df["event_date"].dt.day_name()
    if "venue" not in df.columns:
        df["venue"] = "Unknown"
    if "event_type" not in df.columns:
        df["event_type"] = "Other/Unknown"

    if today is None:
        today = pd.Timestamp.today().normalize()

    df["days_out"] = (df["event_date"].dt.normalize() - df["sale_date"].dt.normalize()).dt.days
    df = df[(df["days_out"] >= 0) & (df["days_out"] <= D_MAX)]
    df["d_bin"] = (df["days_out"] // BIN) * BIN
    df["_today"] = today

    return df


def _cum_series(df: pd.DataFrame) -> pd.DataFrame:
    """Per-event cumulative qty and final totals; cum_pct for HISTORICAL events."""
    s = df.sort_values(["event_name", "days_out"], ascending=[True, False]).copy()
    s["cum_qty"] = s.groupby("event_name", sort=False)["qty_sold"].cumsum()

    finals = s.groupby("event_name", as_index=True)["cum_qty"].max().rename("final_qty")
    s = s.join(finals, on="event_name")

    # historical % (relative to each event's own final)
    s["hist_cum_pct"] = 100 * s["cum_qty"] / s["final_qty"].replace({0: np.nan})

    return s


def build_cohort_library(
    df: pd.DataFrame,
    cohort_tiers: List[List[str]] = DEFAULT_COHORT_TIERS,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build two libraries:
      1) MEDIANS: median hist_cum_pct by (tier..., d_bin) with sample size n
      2) FINALS:  median final_qty by (tier...) to serve as baseline final for upcoming events
    """
    s = _cum_series(df)

    # past events end strictly before "today"
    today = s["_today"].iloc[0]
    hist = s[s["event_date"] < today].copy()

    if hist.empty:
        # Return empty DataFrames with correct structure
        return (
            pd.DataFrame(columns=["tier", "d_bin", "median_pct", "n"]),
            pd.DataFrame(columns=["tier", "median_final", "n_events"]),
        )

    frames = []
    finals = []

    for tier in cohort_tiers:
        keys = tier + ["d_bin"] if tier else ["d_bin"]
        g = (
            hist.groupby(keys, as_index=False)
            .agg(median_pct=("hist_cum_pct", "median"), n=("event_name", "nunique"))
        )
        g["tier"] = "|".join(tier) if tier else "all"
        frames.append(g)

        # median final size per cohort (no d_bin)
        fk = tier if tier else []
        if fk:
            f = (
                hist.groupby(fk, as_index=False)
                .agg(median_final=("final_qty", "median"), n_events=("event_name", "nunique"))
            )
        else:
            f = pd.DataFrame(
                {
                    "median_final": [hist["final_qty"].median()],
                    "n_events": [hist["event_name"].nunique()],
                }
            )
        f["tier"] = "|".join(tier) if tier else "all"
        finals.append(f)

    medians = pd.concat(frames, ignore_index=True)
    finals_lib = pd.concat(finals, ignore_index=True)

    return medians, finals_lib


def _pick_cohort(
    medians: pd.DataFrame,
    finals: pd.DataFrame,
    row: pd.Series,
    d_bin: int,
) -> Tuple[str, float, int, float, int]:
    """
    Choose the most specific cohort with n>=MIN_COHORT.
    Returns: (tier, median_pct_at_d, n, baseline_final, n_events_for_final)
    """
    for tier in DEFAULT_COHORT_TIERS:
        tier_name = "|".join(tier) if tier else "all"

        if tier:
            # Filter medians
            mask = (medians["tier"] == tier_name) & (medians["d_bin"] == d_bin)
            for k in tier:
                if k in medians.columns:
                    mask &= medians[k] == row.get(k, "")
            hit = medians[mask]

            # Finals
            fmask = finals["tier"] == tier_name
            for k in tier:
                if k in finals.columns:
                    fmask &= finals[k] == row.get(k, "")
            fhit = finals[fmask]
        else:
            hit = medians[(medians["tier"] == "all") & (medians["d_bin"] == d_bin)]
            fhit = finals[finals["tier"] == "all"]

        if not hit.empty and int(hit["n"].iloc[0]) >= MIN_COHORT and not fhit.empty:
            return (
                tier_name,
                float(hit["median_pct"].iloc[0]),
                int(hit["n"].iloc[0]),
                float(fhit["median_final"].iloc[0]),
                int(fhit["n_events"].iloc[0]),
            )

    # Last resort: whatever exists
    any_hit = medians[medians["d_bin"] == d_bin]
    any_fin = finals.iloc[[0]] if not finals.empty else pd.DataFrame({"median_final": [1], "n_events": [0]})

    if any_hit.empty:
        return (
            "all",
            0.0,
            0,
            float(any_fin["median_final"].iloc[0]),
            int(any_fin["n_events"].iloc[0]),
        )

    return (
        "all",
        float(any_hit["median_pct"].median()),
        int(any_hit["n"].sum()),
        float(any_fin["median_final"].iloc[0]),
        int(any_fin["n_events"].iloc[0]),
    )


def pace_status(gap: float) -> str:
    """Determine status from gap in percentage points."""
    if np.isnan(gap):
        return "No benchmark"
    if gap <= -AHEAD_PP:
        return "Behind"
    if gap >= AHEAD_PP:
        return "Ahead"
    return "On pace"


def build_watchlist(df_raw: pd.DataFrame, today: Optional[pd.Timestamp] = None) -> Tuple[pd.DataFrame, dict, set]:
    """
    Produce the watchlist table with plain-English column names:
      Event | Days to show | Sold so far (%) | Typical by now (%) | Ahead/behind (pts) | Status | Tickets sold | Comparison group

    Logic:
      - Sold so far (%) = min(100, sold_so_far / baseline_final_from_cohort * 100)
      - Typical by now (%) = cohort median cum% at the same days-out bin
      - Ahead/behind (pts) = Sold so far − Typical by now; Status via ±5 pts
      - Exclude events where Sold so far ≥ 98%

    Returns:
      - watch: DataFrame with watchlist
      - summary: dict with "behind", "evaluated" counts
      - fallback_tiers: set of tier names used
    """
    if today is None:
        today = pd.Timestamp.today().normalize()

    df = _prep(df_raw, today)

    empty_cols = [
        "event",
        "days_out",
        "sold_so_far_pct",
        "typical_at_day_pct",
        "gap_pp",
        "tickets_so_far",
        "status",
        "cohort",
    ]

    if df.empty:
        return pd.DataFrame(columns=empty_cols), {"behind": 0, "evaluated": 0}, set()

    medians, finals_lib = build_cohort_library(df)

    # Aggregate tickets so far per event for upcoming events
    up = df[df["event_date"] >= today].copy()
    if up.empty:
        return pd.DataFrame(columns=empty_cols), {"behind": 0, "evaluated": 0}, set()

    # Get event label if available
    label_col = "event_label" if "event_label" in up.columns else "event_name"

    sold_now = (
        up.groupby(
            ["event_name", "event_date", "event_type", "weekday", "venue"],
            as_index=False,
        )
        .agg(
            tickets_so_far=("qty_sold", "sum"),
            days_out=("days_out", "min"),
            event_label=(label_col, "first"),
        )
    )
    sold_now["d_bin"] = (sold_now["days_out"] // BIN) * BIN

    rows = []
    fallback_tiers: set = set()
    summary = {"behind": 0, "evaluated": 0}

    for _, r in sold_now.iterrows():
        tier, median_at_d, n, baseline_final, n_final = _pick_cohort(
            medians, finals_lib, r, int(r["d_bin"])
        )
        fallback_tiers.add(tier)

        # Guard rails
        baseline_final = max(baseline_final, 1.0)
        tickets_so_far = float(r["tickets_so_far"])
        cum_now = np.clip(100.0 * tickets_so_far / baseline_final, 0, 100)
        gap = cum_now - median_at_d
        status = pace_status(gap)

        summary["evaluated"] += 1
        if status == "Behind":
            summary["behind"] += 1

        # Build readable cohort label
        cohort_label = _format_cohort_label(tier, n, r)

        rows.append(
            {
                "event": r["event_label"] if pd.notna(r["event_label"]) else r["event_name"],
                "days_out": int(r["days_out"]),
                "sold_so_far_pct": round(cum_now, 1),
                "typical_at_day_pct": round(median_at_d, 1),
                "gap_pp": round(gap, 1),
                "tickets_so_far": int(round(tickets_so_far)),
                "status": status,
                "cohort": cohort_label,
            }
        )

    watch = pd.DataFrame(rows)

    if watch.empty:
        return watch, summary, fallback_tiers

    # Hide sold-out / nearly sold-out rows from the action list (≥98%)
    watch = watch[watch["sold_so_far_pct"] < HIDE_SOLDOUT_AT].copy()

    # Rank by most at-risk first (lowest gap), then by days-out
    watch = watch.sort_values(["gap_pp", "days_out"], ascending=[True, True]).reset_index(drop=True)

    # Top 50
    watch = watch.head(50)

    return watch, summary, fallback_tiers


def get_global_pacing_curve(df_raw: pd.DataFrame, today: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """
    Get the global pacing curve (median + IQR) for the booking window chart.
    
    FIXED: Clamps early purchases (>120 days out) into the 120-day bin so the curve
    properly ends at 100% at day 0.
    
    Returns DataFrame with d_bin, median_pct, p25_pct, p75_pct, n (sample size).
    """
    if today is None:
        today = pd.Timestamp.today().normalize()

    # Prepare data - but DON'T filter out early sales yet
    df = df_raw.copy()
    df["sale_date"] = pd.to_datetime(df["sale_date"])
    df["event_date"] = pd.to_datetime(df["event_date"])
    
    # Only historical events
    hist = df[df["event_date"] < today].copy()
    if hist.empty:
        return pd.DataFrame(columns=["d_bin", "median_pct", "p25_pct", "p75_pct", "n"])
    
    # Calculate days_out (can be negative for sales after event, or >120 for early sales)
    hist["days_out"] = (hist["event_date"].dt.normalize() - hist["sale_date"].dt.normalize()).dt.days
    
    # Only keep sales on or before event day
    hist = hist[hist["days_out"] >= 0].copy()
    
    # CLAMP early sales (>120 days) into the 120-day bin so curve reaches 100%
    hist["d_bin"] = np.minimum(hist["days_out"], D_MAX)
    hist["d_bin"] = (hist["d_bin"] // BIN) * BIN
    
    # Build per-event cumulative curves
    # Group by event and bin, sum tickets
    event_bin_sales = (
        hist.groupby(["event_name", "event_date", "d_bin"], as_index=False)["qty_sold"].sum()
    )
    
    # Pivot to wide format: rows = events, columns = bins
    bins = list(range(0, D_MAX + 1, BIN))
    wide = (
        event_bin_sales.pivot_table(
            index=["event_name", "event_date"],
            columns="d_bin",
            values="qty_sold",
            aggfunc="sum",
            fill_value=0,
        )
        .reindex(columns=bins, fill_value=0)
    )
    
    # Cumulative from 120 → 0 (reverse order, then cumsum)
    rev_cols = bins[::-1]  # [120, 117, ..., 3, 0]
    cum_rev = wide[rev_cols].cumsum(axis=1)
    
    # Calculate percentage of final total
    totals = wide.sum(axis=1).replace(0, np.nan)
    pct_rev = (cum_rev.T / totals).T * 100.0
    
    # Compute stats across events for each bin
    curve = pd.DataFrame({
        "d_bin": rev_cols,
        "median_pct": pct_rev.median(axis=0).values,
        "p25_pct": pct_rev.quantile(0.25, axis=0).values,
        "p75_pct": pct_rev.quantile(0.75, axis=0).values,
        "n": pct_rev.notna().sum(axis=0).values,
    })
    
    # Fill NaN and enforce monotonicity
    for col in ("p25_pct", "median_pct", "p75_pct"):
        curve[col] = curve[col].ffill().fillna(0)
        curve[col] = curve[col].cummax().clip(0, 100)
    
    # Ensure p25 ≤ median ≤ p75
    curve["p25_pct"] = curve["p25_pct"].clip(upper=curve["median_pct"])
    curve["p75_pct"] = curve["p75_pct"].clip(lower=curve["median_pct"])
    
    return curve.reset_index(drop=True)

