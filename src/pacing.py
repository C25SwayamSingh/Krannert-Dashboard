"""
Pacing module for the Krannert dashboard.

The benchmark is SELF-CONSISTENT: for every historical event in a cohort we
compute the exact same statistic the watchlist computes for an upcoming event —
tickets sold by day D divided by the cohort's median final sales, carried
forward to every days-out bin — and "Typical by now" is the median of that
across cohort events. By construction the median historical event paces at
gap 0, so Behind/Ahead is a genuinely relative signal.

Two earlier biases this design removes:
  - The old benchmark took the median over transaction ROWS present at a bin,
    so quiet events dropped out of their own baseline and busy/early-selling
    events dominated it (inflating "typical by now", especially far out).
  - Upcoming events were normalized by the cohort median final while the
    benchmark was normalized by each event's OWN final, so smaller-than-median
    events could never look on pace.
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
    df = df[df["days_out"] >= 0]
    # Clamp early purchases (>D_MAX days out) into the outermost bin instead of
    # dropping them, so cumulative curves account for all pre-event sales.
    df["d_bin"] = (np.minimum(df["days_out"], D_MAX) // BIN) * BIN
    df["_today"] = today

    return df


# The watchlist tracks each (event, event_type) slice as its own row, so the
# historical cohort curves must be built at the same unit of analysis.
UNIT_KEYS = ["event_name", "event_type"]


def _event_curves(hist: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Per-unit cumulative tickets carried forward to EVERY days-out bin, where a
    unit is an (event_name, event_type) slice — the same grain as the watchlist.

    Returns:
      - cum: DataFrame indexed by UNIT_KEYS, one column per d_bin (D_MAX..0),
        holding tickets sold by that bin (0 for bins with no activity yet)
      - finals: Series of total tickets per unit
      - meta: DataFrame of cohort keys (event_type, weekday, venue) per unit
    """
    bins = list(range(0, D_MAX + 1, BIN))
    eb = hist.groupby(UNIT_KEYS + ["d_bin"], as_index=False)["qty_sold"].sum()
    wide = (
        eb.pivot_table(
            index=UNIT_KEYS,
            columns="d_bin",
            values="qty_sold",
            aggfunc="sum",
            fill_value=0,
        )
        .reindex(columns=bins, fill_value=0)
    )
    cum = wide[bins[::-1]].cumsum(axis=1)
    finals = wide.sum(axis=1).rename("final_qty")
    meta = hist.groupby(UNIT_KEYS).agg(
        weekday=("weekday", "first"),
        venue=("venue", "first"),
    )
    meta["event_type"] = meta.index.get_level_values("event_type")
    # Rename index levels so "event_type" is unambiguous as a groupby column
    meta.index = meta.index.set_names(["unit_event_name", "unit_event_type"])
    return cum, finals, meta


def build_cohort_library(
    df: pd.DataFrame,
    cohort_tiers: List[List[str]] = DEFAULT_COHORT_TIERS,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build two libraries:
      1) MEDIANS: per (tier..., d_bin), the median across cohort events of
         min(100, tickets_by_bin / cohort_median_final * 100) — the same
         statistic the watchlist computes for upcoming events, so the median
         historical event paces at gap 0 by construction
      2) FINALS:  median final tickets per cohort, the baseline denominator
         for upcoming events
    """
    today = df["_today"].iloc[0] if not df.empty else pd.Timestamp.today().normalize()
    hist = df[df["event_date"] < today]

    if hist.empty:
        # Return empty DataFrames with correct structure
        return (
            pd.DataFrame(columns=["tier", "d_bin", "median_pct", "n"]),
            pd.DataFrame(columns=["tier", "median_final", "n_events"]),
        )

    cum, finals, meta = _event_curves(hist)

    frames = []
    fin_rows = []

    for tier in cohort_tiers:
        tier_name = "|".join(tier) if tier else "all"

        if tier:
            groups = meta.groupby(tier, dropna=False).groups.items()
        else:
            groups = [(None, meta.index)]

        for gkey, names in groups:
            median_final = max(float(finals.loc[names].median()), 1.0)
            norm = (cum.loc[names] / median_final * 100.0).clip(0, 100)
            med = norm.median(axis=0)

            g = pd.DataFrame({"d_bin": med.index.astype(int), "median_pct": med.values})
            g["n"] = len(names)
            g["tier"] = tier_name

            f = {"tier": tier_name, "median_final": median_final, "n_events": len(names)}

            if tier:
                key_vals = gkey if isinstance(gkey, tuple) else (gkey,)
                for k, v in zip(tier, key_vals):
                    g[k] = v
                    f[k] = v

            frames.append(g)
            fin_rows.append(f)

    medians = pd.concat(frames, ignore_index=True)
    finals_lib = pd.DataFrame(fin_rows)

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
        "tickets_at_risk",
        "status",
        "cohort",
    ]

    if df.empty:
        return pd.DataFrame(columns=empty_cols), {"behind": 0, "on_pace": 0, "ahead": 0, "evaluated": 0}, set()

    medians, finals_lib = build_cohort_library(df)

    # Aggregate tickets so far per event for upcoming events
    up = df[df["event_date"] >= today].copy()
    if up.empty:
        return pd.DataFrame(columns=empty_cols), {"behind": 0, "on_pace": 0, "ahead": 0, "evaluated": 0}, set()

    # Get event label if available
    label_col = "event_label" if "event_label" in up.columns else "event_name"

    sold_now = (
        up.groupby(
            ["event_name", "event_date", "event_type", "weekday", "venue"],
            as_index=False,
        )
        .agg(
            tickets_so_far=("qty_sold", "sum"),
            event_label=(label_col, "first"),
        )
    )
    # Days to show is measured from "today", not from the most recent sale —
    # otherwise events with stale sales get benchmarked at the wrong bin.
    sold_now["days_out"] = (sold_now["event_date"].dt.normalize() - today).dt.days
    sold_now = sold_now[sold_now["days_out"] <= D_MAX]
    if sold_now.empty:
        return pd.DataFrame(columns=empty_cols), {"behind": 0, "on_pace": 0, "ahead": 0, "evaluated": 0}, set()
    sold_now["d_bin"] = (sold_now["days_out"] // BIN) * BIN

    rows = []
    fallback_tiers: set = set()
    summary = {"behind": 0, "on_pace": 0, "ahead": 0, "evaluated": 0}

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
        elif status == "Ahead":
            summary["ahead"] += 1
        elif status == "On pace":
            summary["on_pace"] += 1

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
                "tickets_at_risk": round(max(0.0, -gap) * baseline_final / 100.0, 1),
                "status": status,
                "cohort": cohort_label,
            }
        )

    watch = pd.DataFrame(rows)

    if watch.empty:
        return watch, summary, fallback_tiers

    # Hide sold-out / nearly sold-out rows from the action list (≥98%)
    watch = watch[watch["sold_so_far_pct"] < HIDE_SOLDOUT_AT].copy()

    # Rank by tickets at risk (gap × typical audience size) so material
    # shortfalls outrank 1-2 ticket micro-slices with huge percentage gaps,
    # breaking ties by lowest gap, then by days-out
    watch = watch.sort_values(
        ["tickets_at_risk", "gap_pp", "days_out"], ascending=[False, True, True]
    ).reset_index(drop=True)

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

