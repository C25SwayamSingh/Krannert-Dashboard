"""
Krannert Dashboard — Streamlit application
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import data_prep
from src import pacing
from src.figs_collab import (
    fig_categories_pre,
    fig_categories_post,
    fig_top_categories_pre_post,
    fig_top_events_pre_post,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent / "data"
ASSETS_DIR = Path(__file__).parent / "assets"
DEFAULT_DATA_PATH = DATA_DIR / "sales_2016_2026_combined.csv"

MAX_FILTER_VALUES = 3
LEAD_BUCKET_LABELS = [bucket[2] for bucket in data_prep.LEAD_BUCKETS]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def inject_styles() -> None:
    """Inject global CSS for the warm, minimal theme."""
    st.markdown(
        """
        <style>
        /* Global layout */
        .block-container {
            max-width: 1300px;
            padding-top: 1.5rem;
            padding-bottom: 3rem;
        }
        body {
            background: #F4EEE6;
            color: #1F1A17;
        }
        .stApp {
            background: #F4EEE6;
        }
        /* Cards */
        .card {
            background: #FFFFFF;
            border-radius: 14px;
            padding: 14px 18px 12px;
            box-shadow: 0 2px 6px rgba(47,42,36,0.06);
            border: 1px solid rgba(47,42,36,0.08);
            margin-bottom: 12px;
        }
        .card h3, .card h4, .card h5 {
            margin-top: 0.1rem;
        }
        /* KPI tiles */
        .kpi-card {
            padding: 14px 16px 12px;
        }
        .kpi-label {
            font-size: 12px;
            letter-spacing: 0.6px;
            text-transform: uppercase;
            color: #5b524a;
            margin-bottom: 2px;
        }
        .kpi-value {
            font-size: 26px;
            font-weight: 700;
            margin-bottom: 2px;
            color: #1F1A17;
        }
        .kpi-delta {
            font-size: 12px;
            color: #6b625a;
        }
        /* KPI help icon (fallback tooltip) */
        .kpi-help {
            cursor: help;
            font-size: 0.9rem;
            display: inline-block;
            padding-left: 6px;
            line-height: 1;
            opacity: 0.85;
        }
        /* Buttons */
        .primary-btn button, .primary-btn>button {
            background: #2F2A24 !important;
            color: #FFFFFF !important;
            border-radius: 999px !important;
            border: 1px solid #2F2A24 !important;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .primary-btn button:hover { transform: translateY(-1px); }
        .ghost-btn button, .ghost-btn>button {
            background: transparent !important;
            color: #2F2A24 !important;
            border-radius: 999px !important;
            border: 1px solid rgba(47,42,36,0.3) !important;
        }
        /* Badges & helper text */
        .pill {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 10px;
            background: #F4EEE6;
            border: 1px solid rgba(47,42,36,0.12);
            color: #2F2A24;
            font-size: 11px;
        }
        .muted {
            color: #6b625a;
            font-size: 13px;
        }
        /* Hide Streamlit settings (theme chooser) to keep a single enforced theme */
        div[data-testid="stToolbar"] button[title="Settings"] {
            display: none !important;
        }
        /* Compact, readable help list */
        .help-list {
            margin: 0.25rem 0 0.5rem 0;
            padding-left: 1.0rem;
            line-height: 1.6;
            color: var(--text-color, #1F1A17);
            font-size: 14px;
        }
        .help-list li {
            margin: 0.2rem 0;
        }
        /* Keep expander body aligned with card style */
        [data-testid="stExpander"] .streamlit-expanderContent {
            background: rgba(255,255,255,0.6);
            border-radius: 10px;
            padding: 0.5rem 0.75rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def label_with_help(text: str, help_text: str) -> None:
    """Render a label with a small info icon and instant CSS tooltip."""
    # Escape quotes in help text for HTML attribute
    safe_help = help_text.replace('"', '&quot;').replace("'", "&#39;")
    st.markdown(
        f"""
        <style>
            .kpi-tooltip-wrap {{
                position: relative;
                display: inline-flex;
                align-items: center;
            }}
            .kpi-tooltip-icon {{
                cursor: pointer;
                display: inline-flex;
                align-items: center;
                justify-content: center;
                width: 16px;
                height: 16px;
                border-radius: 50%;
                background: #e0dcd6;
                color: #5b524a;
                font-size: 11px;
                font-weight: 600;
                font-style: italic;
                font-family: Georgia, serif;
            }}
            .kpi-tooltip-icon:hover + .kpi-tooltip-text,
            .kpi-tooltip-text:hover {{
                visibility: visible;
                opacity: 1;
            }}
            .kpi-tooltip-text {{
                visibility: hidden;
                opacity: 0;
                position: absolute;
                left: 0;
                bottom: 100%;
                margin-bottom: 6px;
                background: #2F2A24;
                color: #fff;
                padding: 8px 12px;
                border-radius: 6px;
                font-size: 12px;
                font-weight: 400;
                font-style: normal;
                width: 260px;
                z-index: 1000;
                box-shadow: 0 2px 8px rgba(0,0,0,0.15);
                transition: opacity 0.1s ease-in-out;
                line-height: 1.4;
            }}
        </style>
        <div style="display: flex; align-items: center; margin-bottom: 4px;">
            <strong style="margin-right: 6px;">{text}</strong>
            <span class="kpi-tooltip-wrap">
                <span class="kpi-tooltip-icon">i</span>
                <span class="kpi-tooltip-text">{safe_help}</span>
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )


def _format_int(value: float | int | None) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "–"
    return f"{value:,.0f}"


def _format_currency(value: float | int | None) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "–"
    return f"${value:,.0f}"


def _format_percent(value: float | None, decimals: int = 1) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "–"
    return f"{value:.{decimals}f}%"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_dataset(uploaded_file, local_path: str | Path) -> tuple[pd.DataFrame | None, str]:
    if uploaded_file is not None:
        try:
            return data_prep.load_csv(uploaded_file), uploaded_file.name
        except Exception as exc:
            st.error(f"Unable to read uploaded file: {exc}")
            return None, uploaded_file.name

    path = Path(local_path).expanduser()
    if not path.exists():
        st.warning(f"Local CSV not found at {path}. Update the path or upload a file.")
        return None, str(path)

    try:
        return data_prep.load_csv(path), str(path)
    except Exception as exc:
        st.error(f"Unable to read {path}: {exc}")
        return None, str(path)


def render_hero(preloaded_path: Path) -> None:
    """Hero section with minimal copy; uploads handled in sidebar."""
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Krannert Sales Insights")
    st.markdown(
        '<div class="muted">Track booking pace, heatmap demand, and pre/post shifts. '
        "Use the sidebar to swap CSVs.</div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------
def summarize_filters(filters: dict) -> str:
    def summarize(values: list[str], label: str) -> str:
        if not values:
            return f"{label}: All"
        snippet = ", ".join(values[:MAX_FILTER_VALUES])
        remainder = len(values) - MAX_FILTER_VALUES
        if remainder > 0:
            snippet += f" +{remainder} more"
        return f"{label}: {snippet}"

    seasons_text = summarize(filters.get("seasons") or [], "Season")
    channels_text = summarize(filters.get("channels") or [], "Channel")

    start, end = filters.get("date_range", (None, None))
    if start is not None and end is not None:
        date_text = f"Dates: {start.date():%b %d %Y} → {end.date():%b %d %Y}"
    else:
        date_text = "Dates: All"

    return " | ".join([seasons_text, channels_text, date_text])


def default_filter_bounds(df: pd.DataFrame) -> dict:
    """Compute sensible default filter values."""
    today = pd.Timestamp.today().normalize()
    defaults = {}

    # Default seasons = last 5
    if "season" in df.columns:
        seasons = sorted(df["season"].dropna().unique())
        defaults["all_seasons"] = seasons
        defaults["selected_seasons"] = seasons[-5:] if len(seasons) > 5 else seasons
    else:
        defaults["all_seasons"] = []
        defaults["selected_seasons"] = []

    # Event date range = [2016-01-01, today+365]
    min_event = pd.Timestamp("2016-01-01")
    if df["event_date"].notna().any():
        data_max = df["event_date"].max()
        max_event = min(data_max, today + pd.Timedelta(days=365))
    else:
        max_event = today + pd.Timedelta(days=365)
    defaults["date_bounds"] = (min_event.date(), max_event.date())
    return defaults


def sidebar_filters(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, str]:
    defaults = default_filter_bounds(df)
    st.sidebar.subheader("Filters")

    seasons = defaults["all_seasons"]
    selected_seasons = st.sidebar.multiselect(
        "Season", seasons, default=defaults["selected_seasons"], placeholder="All seasons"
    )

    channels = sorted(df["channel"].dropna().unique()) if "channel" in df.columns else []
    selected_channels = st.sidebar.multiselect(
        "Channel", channels, default=channels, placeholder="All channels"
    )

    min_date, max_date = defaults["date_bounds"]
    date_range = st.sidebar.date_input(
        "Event date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    if isinstance(date_range, tuple):
        start_date, end_date = date_range
    else:
        start_date = end_date = date_range

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    mask = pd.Series(True, index=df.index)
    if selected_seasons:
        mask &= df["season"].isin(selected_seasons)
    if selected_channels and "channel" in df.columns:
        mask &= df["channel"].isin(selected_channels)
    mask &= df["event_date"].between(start_ts, end_ts)

    filtered = df.loc[mask].copy()
    filters = {
        "seasons": selected_seasons,
        "channels": selected_channels,
        "date_range": (start_ts, end_ts),
    }
    return filtered, filters, summarize_filters(filters)


# ---------------------------------------------------------------------------
# KPIs
# ---------------------------------------------------------------------------
def compute_kpis(base: pd.DataFrame, watch_summary: dict, today: pd.Timestamp) -> dict:
    """Compute time-boxed KPIs: last 30 days and next 30 days."""

    # Last 30 days
    last_30_mask = (base["sale_date"] >= today - pd.Timedelta(days=29)) & (base["sale_date"] <= today)
    prev_30_mask = (base["sale_date"] >= today - pd.Timedelta(days=59)) & (
        base["sale_date"] < today - pd.Timedelta(days=29)
    )

    tickets_30d = float(base.loc[last_30_mask, "qty_sold"].sum())
    tickets_prev = float(base.loc[prev_30_mask, "qty_sold"].sum())
    delta_30d = ((tickets_30d - tickets_prev) / tickets_prev) if tickets_prev > 0 else None

    # Avg price last 30 days
    if "amount" in base.columns:
        amount_30d = base.loc[last_30_mask, "amount"].sum()
        qty_30d = base.loc[last_30_mask, "qty_sold"].sum()
        avg_price_30d = amount_30d / qty_30d if qty_30d > 0 else None
    else:
        avg_price_30d = None

    # Open events next 30 days
    open_events = base[
        (base["event_date"] >= today) & (base["event_date"] <= today + pd.Timedelta(days=30))
    ]["event_instance"].nunique()

    # % behind pace (events ≤120d with gap_pp <= -5)
    evaluated = watch_summary.get("evaluated", 0)
    behind = watch_summary.get("behind", 0)
    behind_pct = (behind / evaluated) * 100 if evaluated > 0 else 0

    return {
        "tickets_30d": tickets_30d,
        "delta_30d": delta_30d,
        "avg_price_30d": avg_price_30d,
        "open_events": open_events,
        "behind_pct": behind_pct,
        "behind_detail": f"{behind} of {evaluated}" if evaluated > 0 else "–",
    }


def render_kpis(kpis: dict) -> None:
    col1, col2, col3, col4 = st.columns(4)

    delta_display = None
    if kpis["delta_30d"] is not None:
        delta_display = _format_percent(kpis["delta_30d"] * 100)

    with col1:
        label_with_help(
            "Tickets last 30 days",
            "Tickets sold in the last 30 days. Delta vs the prior 30 days.",
        )
        st.metric(label="", value=_format_int(kpis["tickets_30d"]), delta=delta_display)

    with col2:
        label_with_help(
            "Avg price last 30 days",
            "Average realized price per ticket over the last 30 days.",
        )
        st.metric(label="", value=_format_currency(kpis["avg_price_30d"]))

    with col3:
        label_with_help(
            "Open events next 30 days",
            "Distinct events scheduled in the next 30 days that are still on sale.",
        )
        st.metric(label="", value=_format_int(kpis["open_events"]))

    with col4:
        label_with_help(
            "% events behind pace (≤120d)",
            "Share of upcoming events (≤120 days out) selling slower than their historical median at the same days-out. These are the top marketing/price-review priorities.",
        )
        st.metric(
            label="",
            value=_format_percent(kpis["behind_pct"]),
            delta=kpis["behind_detail"],
            delta_color="inverse",
        )


# ---------------------------------------------------------------------------
# Watchlist table (using new pacing module)
# ---------------------------------------------------------------------------
def render_watchlist_summary(watchlist_df: pd.DataFrame, upcoming_window_days: int = 120) -> None:
    if watchlist_df is None or watchlist_df.empty:
        st.caption(f"Showing **0** upcoming events (≤{upcoming_window_days}d) that match your filters.")
        return

    # Normalize the status labels robustly
    status_col = None
    for c in ["status", "Status", "STATUS"]:
        if c in watchlist_df.columns:
            status_col = c
            break

    if status_col is None:
        # No status column; just show the total
        st.caption(f"Showing **{len(watchlist_df)}** upcoming events (≤{upcoming_window_days}d) that match your filters.")
        return

    s = (watchlist_df[status_col]
         .astype(str).str.strip().str.lower()
         .map({"behind": "Behind", "on pace": "On pace", "ahead": "Ahead"}))

    counts = s.value_counts().reindex(["Behind", "On pace", "Ahead"], fill_value=0)
    n_total = int(counts.sum())
    n_behind = int(counts["Behind"])
    n_on     = int(counts["On pace"])
    n_ahead  = int(counts["Ahead"])

    # Lightweight CSS for compact color pills (keeps current theme)
    # Using .pill-status to avoid conflict with global .pill
    st.markdown("""
        <style>
          .pill-status {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 999px;
            font-size: 0.85rem;
            margin-left: 6px;
          }
          .pill-red { background: #ffe8e6; color: #b42318; }
          .pill-amber { background: #fff3cd; color: #7a5b00; }
          .pill-green { background: #eaf7ec; color: #1a7f37; }
        </style>
    """, unsafe_allow_html=True)

    st.markdown(
        (f"**Showing {n_total} upcoming events (≤{upcoming_window_days}d) that match your filters —** "
         f"<span class='pill-status pill-red'>{n_behind} behind</span> "
         f"<span class='pill-status pill-amber'>{n_on} on pace</span> "
         f"<span class='pill-status pill-green'>{n_ahead} ahead</span>."
        ),
        unsafe_allow_html=True
    )


def render_watchlist(watch_table: pd.DataFrame, fallback_tiers: set, filters_summary: str) -> None:
    """Render the event pacing watchlist with Plotly table for visibility."""
    with st.expander("What do these columns mean?"):
        st.markdown("""
- **Days-out**: Days until the event (e.g., 45 = event is 45 days away).
- **Sold so far (%)**: 100 × tickets sold so far ÷ typical final audience size.
- **Typical at this day (%)**: Median sell-through for similar past events at the same days-out.
- **Gap vs typical (pp)**: *Sold so far – Typical at this day* (percentage points). Negative = behind; positive = ahead.
- **Status**: Ahead / On pace / Behind (on-pace band = −5 to +5 pp).
- **Tickets so far**: Raw ticket count sold to date for the event.
- **Cohort**: Comparison group used (e.g., event type; event type + weekday/+venue; or global when history is thin).
        """)
    
    # Render concise summary
    try:
        window_days = pacing.D_MAX
    except AttributeError:
        window_days = 120
    render_watchlist_summary(watch_table, upcoming_window_days=window_days)

    st.caption(filters_summary)

    if watch_table.empty:
        st.info("No upcoming events in the next 120 days, or not enough historical data to compute pacing benchmarks.")
        return

    if fallback_tiers:
        tiers = {pacing.TIER_LABELS.get(t, t) for t in fallback_tiers}
        st.caption(f"Cohort used: {', '.join(sorted(tiers))}")

    # Column order and configs
    display_cols = [
        "event",
        "days_out",
        "sold_so_far_pct",
        "typical_at_day_pct",
        "gap_pp",
        "tickets_so_far",
        "status",
        "cohort",
    ]
    display = watch_table[[c for c in display_cols if c in watch_table.columns]].copy()

    # Color palette for status
    def status_color(status: str) -> str:
        if status == "Behind":
            return "#ffd6de"  # brighter pink
        if status == "Ahead":
            return "#e3f7e3"  # soft green
        return "#fff7da"      # soft yellow

    # Format values for display
    fmt = {
        "days_out": lambda v: f"{v:d}",
        "sold_so_far_pct": lambda v: f"{v:.1f}%",
        "typical_at_day_pct": lambda v: f"{v:.1f}%",
        "gap_pp": lambda v: f"{v:.1f}",
        "tickets_so_far": lambda v: f"{int(v):,}",
    }
    formatted_cols = []
    for col in display_cols:
        if col in fmt:
            formatted_cols.append(display[col].apply(fmt[col]).tolist())
        else:
            formatted_cols.append(display[col].tolist())

    # Row-wise colors
    row_colors = [[status_color(s) for s in display["status"]]] * len(display_cols)

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=[
                        "<b>Event</b>",
                        "<b>Days-out</b>",
                        "<b>Sold so far (%)</b>",
                        "<b>Typical at this day (%)</b>",
                        "<b>Gap vs typical (pp)</b>",
                        "<b>Tickets so far</b>",
                        "<b>Status</b>",
                        "<b>Cohort</b>",
                    ],
                    fill_color="#2c3e50",
                    font=dict(color="white", size=13),
                    align="left",
                    height=32,
                ),
                cells=dict(
                    values=formatted_cols,
                    fill_color=row_colors,
                    align="left",
                    font=dict(color="#1a1a1a", size=12),
                    height=30,
                ),
            )
        ]
    )
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=min(600, 35 + 30 * len(display) + 60))
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

    # Summary stats with emojis
    behind_count = int((watch_table["status"] == "Behind").sum())
    ahead_count = int((watch_table["status"] == "Ahead").sum())
    on_pace_count = int((watch_table["status"] == "On pace").sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("🔴 Behind pace", behind_count)
    c2.metric("🟨 On pace", on_pace_count)
    c3.metric("🟢 Ahead of pace", ahead_count)


# ---------------------------------------------------------------------------
# Booking window chart
# ---------------------------------------------------------------------------
DAYS_MAX = pacing.D_MAX
BIN_SIZE = pacing.BIN
MIN_EVENTS_BASELINE = 25


def cohort_candidates(filters: dict) -> list:
    """
    Order of specificity: event_type -> global.
    Extend with weekday/venue if you add those filters.
    """
    et = None
    # if a single event_type is selected in filters, use it
    if filters.get("event_types") and len(filters["event_types"]) == 1:
        et = filters["event_types"][0]
    cands = []
    if et:
        cands.append(("event type only", lambda d: d["event_type"] == et))
    # global fallback
    cands.append(("all events (global)", lambda d: d["event_date"] < pd.Timestamp.max))
    return cands


def build_baseline_curve(df_all: pd.DataFrame, today: pd.Timestamp, filters: dict, mode: str) -> tuple[pd.DataFrame, str, int]:
    """
    Build baseline pacing curve with auto-fallback.
    Returns (curve_df, cohort_label, n_events).
    """
    df_past = df_all[df_all["event_date"] < today].copy()
    if df_past.empty:
        return pd.DataFrame(), "no data", 0

    # Respect explicit override to global
    if mode == "Global":
        cands = [("all events (global)", lambda d: d["event_date"] < today)]
    else:
        cands = cohort_candidates(filters)

    chosen_label = "all events (global)"
    chosen_mask = df_past["event_date"] < today
    n_events = 0

    for label, mask_fn in cands:
        mask = mask_fn(df_past)
        n = df_past.loc[mask, ["event_name", "event_date"]].drop_duplicates().shape[0]
        chosen_label, chosen_mask, n_events = label, mask, n
        if n >= MIN_EVENTS_BASELINE or label == "all events (global)":
            break

    d = df_past.loc[chosen_mask].copy()
    d = d.dropna(subset=["event_date", "sale_date", "qty_sold"])
    if d.empty:
        return pd.DataFrame(), chosen_label, int(n_events)

    d["days_out"] = (d["event_date"].dt.normalize() - d["sale_date"].dt.normalize()).dt.days
    d["days_out"] = d["days_out"].clip(lower=0, upper=DAYS_MAX)
    d["d_bin"] = (d["days_out"] // BIN_SIZE) * BIN_SIZE

    # Ensure all bins exist so day 0 reaches 100%
    bins = list(range(0, DAYS_MAX + 1, BIN_SIZE))

    # Pivot to wide per event per bin
    per_event_bin = (
        d.groupby(["event_name", "event_date", "d_bin"])["qty_sold"].sum().reset_index()
    )
    wide = (
        per_event_bin.pivot_table(
            index=["event_name", "event_date"],
            columns="d_bin",
            values="qty_sold",
            aggfunc="sum",
            fill_value=0,
        )
        .reindex(columns=bins, fill_value=0)
    )

    # Cumulative from farthest (120) toward 0
    rev_cols = bins[::-1]
    cum_rev = wide[rev_cols].cumsum(axis=1)
    totals = wide.sum(axis=1).replace(0, np.nan)
    pct_rev = (cum_rev.T / totals).T  # 0..1
    pct_rev = pct_rev[rev_cols]  # keep 120 -> 0

    agg = pd.DataFrame(
        {
            "d_bin": rev_cols,
            "median_pct": pct_rev.median(axis=0).values,
            "p25": pct_rev.quantile(0.25, axis=0).values,
            "p75": pct_rev.quantile(0.75, axis=0).values,
            "n": pct_rev.notna().sum(axis=0).values,
        }
    )

    # Ensure monotonic rising toward day 0
    for col in ("p25", "median_pct", "p75"):
        agg[col] = agg[col].ffill().fillna(0)
        agg[col] = agg[col].cummax().clip(0, 1)

    # convert to percentages
    for col in ("p25", "median_pct", "p75"):
        agg[col] = (agg[col] * 100).clip(0, 100)

    # rename to expected columns
    agg = agg.rename(columns={"p25": "p25_pct", "p75": "p75_pct"})

    return agg, chosen_label, int(n_events)


def build_current_curve(df_all: pd.DataFrame, today: pd.Timestamp) -> pd.DataFrame | None:
    """
    Build the current selection overlay curve for upcoming events.
    """
    fut = df_all[df_all["event_date"] >= today].copy()
    if fut.empty:
        return None
    fut = fut.dropna(subset=["event_date", "sale_date", "qty_sold"])
    fut["days_out"] = (fut["event_date"].dt.normalize() - fut["sale_date"].dt.normalize()).dt.days
    fut = fut[(fut["days_out"] >= 0) & (fut["days_out"] <= DAYS_MAX)]
    if fut.empty:
        return None
    fut["d_bin"] = (fut["days_out"] // BIN_SIZE) * BIN_SIZE

    # tickets per event per bin
    per_event = fut.groupby(["event_name", "event_date", "d_bin"])["qty_sold"].sum().unstack(fill_value=0)
    # need at least 3 upcoming events to display
    if per_event.shape[0] < 3:
        return None
    rev_cols = sorted(per_event.columns, reverse=True)
    cum_rev = per_event[rev_cols].cumsum(axis=1)
    totals = per_event.sum(axis=1).replace(0, np.nan)
    pct_rev = (cum_rev.T / totals).T * 100.0
    pct_rev = pct_rev[rev_cols]
    mean_curve = pct_rev.mean(axis=0, skipna=True).reset_index()
    mean_curve.columns = ["d_bin", "cum_pct"]
    return mean_curve.sort_values("d_bin", ascending=False).reset_index(drop=True)


def booking_window_fig(global_curve: pd.DataFrame, current_curve: pd.DataFrame | None = None) -> go.Figure:
    """
    Booking window pacing chart with X axis 120 → 0.
    Now ends at 100% at day 0 (fixed clamping of early sales).
    """
    fig = go.Figure()

    if global_curve.empty:
        fig.update_layout(title="Pacing curve unavailable (no historical data)", template="plotly_white")
        return fig

    # IQR band (filled area between p25 and p75)
    fig.add_trace(
        go.Scatter(
            x=list(global_curve["d_bin"]) + list(global_curve["d_bin"][::-1]),
            y=list(global_curve["p75_pct"]) + list(global_curve["p25_pct"][::-1]),
            fill="toself",
            mode="lines",
            line=dict(width=0),
            fillcolor="rgba(31,119,180,0.15)",
            name="Historic IQR (p25–p75)",
            hoverinfo="skip",
        )
    )

    # Median line
    fig.add_trace(
        go.Scatter(
            x=global_curve["d_bin"],
            y=global_curve["median_pct"],
            mode="lines",
            line=dict(color="#1f77b4", width=3),
            name="Historic median",
            hovertemplate="Day %{x}: %{y:.1f}%<extra></extra>",
        )
    )

    # Overlay current selection if provided
    if current_curve is not None and not current_curve.empty:
        fig.add_trace(
            go.Scatter(
                x=current_curve["d_bin"],
                y=current_curve["cum_pct"],
                mode="lines",
                line=dict(color="#ff7f0e", width=3, dash="dash"),
                name="Current selection",
                hovertemplate="Day %{x}: %{y:.1f}%<extra></extra>",
            )
        )

    # D-30 and D-7 checkpoint lines
    for d, label in [(30, "D-30"), (7, "D-7")]:
        fig.add_vline(
            x=d,
            line_dash="dot",
            line_color="gray",
            opacity=0.5,
            annotation_text=label,
            annotation_position="top",
        )

    fig.update_layout(
        template="plotly_white",
        legend_orientation="h",
        legend_yanchor="bottom",
        legend_y=1.02,
        legend_x=0,
        margin=dict(l=10, r=10, t=40, b=10),
    )

    # Y axis 0-100%
    fig.update_yaxes(
        title_text="Cumulative % of tickets sold (of final)",
        range=[0, 105],
        ticksuffix="%",
        tickformat=".0f",
        rangemode="tozero",
    )

    # X axis inverted: 120 → 0
    fig.update_xaxes(
        title_text="Days before event (120 → 0)",
        autorange="reversed",
        range=[pacing.D_MAX, 0],
        dtick=15,
    )

    return fig


# ---------------------------------------------------------------------------
# Simple visuals (season totals, monthly distribution, trend, heatmap)
# ---------------------------------------------------------------------------
def _ensure_dates(df: pd.DataFrame) -> pd.DataFrame:
    z = df.copy()
    if "sale_date" in z.columns:
        z["sale_date"] = pd.to_datetime(z["sale_date"], errors="coerce")
    if "event_date" in z.columns:
        z["event_date"] = pd.to_datetime(z["event_date"], errors="coerce")
    return z


def total_tickets_by_season_fig(df: pd.DataFrame) -> go.Figure | None:
    if df.empty or "season" not in df.columns:
        return None
    z = _ensure_dates(df)
    s = (
        z.groupby("season", dropna=False)["qty_sold"]
        .sum()
        .reset_index()
        .rename(columns={"qty_sold": "tickets"})
    )
    if "season" in s:
        try:
            s["start"] = s["season"].astype(str).str.slice(0, 4).astype(int)
            s = s.sort_values("start")
        except Exception:
            pass
    fig = px.bar(s, x="season", y="tickets", title="Total tickets by season", text="tickets")
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(yaxis_title="Tickets", xaxis_title="Season", margin=dict(l=10, r=10, t=40, b=20))
    fig.update_yaxes(tickformat=",")
    return fig


def monthly_distribution_fig(df: pd.DataFrame) -> go.Figure | None:
    if df.empty:
        return None
    z = _ensure_dates(df)
    if "sale_date" not in z.columns:
        return None
    z["month"] = z["sale_date"].dt.to_period("M").dt.to_timestamp()
    monthly = z.groupby("month", as_index=False)["qty_sold"].sum()
    if monthly.empty:
        return None
    fig = px.histogram(monthly, x="qty_sold", nbins=30, title="Distribution of monthly tickets (all years)")
    fig.update_layout(xaxis_title="Tickets per month", yaxis_title="# of months", margin=dict(l=10, r=10, t=40, b=20))
    fig.update_xaxes(tickformat=",")
    return fig


def monthly_trend_fig(df: pd.DataFrame) -> go.Figure | None:
    if df.empty:
        return None
    z = _ensure_dates(df)
    if "sale_date" not in z.columns:
        return None
    z["month"] = z["sale_date"].dt.to_period("M").dt.to_timestamp()
    monthly = z.groupby("month", as_index=False)["qty_sold"].sum().sort_values("month")
    if monthly.empty:
        return None
    monthly["roll3"] = monthly["qty_sold"].rolling(3, center=True).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=monthly["month"], y=monthly["qty_sold"], mode="lines", name="Monthly total", line=dict(width=1)))
    fig.add_trace(go.Scatter(x=monthly["month"], y=monthly["roll3"], mode="lines", name="3-mo avg", line=dict(width=2)))
    fig.add_shape(type="rect", x0="2020-03-01", x1="2021-06-30", y0=0, y1=1, yref="paper", xref="x", fillcolor="grey", opacity=0.12, line_width=0)
    fig.update_layout(
        title="Monthly tickets sold",
        yaxis_title="Tickets",
        xaxis_title="Month",
        margin=dict(l=10, r=10, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig.update_yaxes(tickformat=",")
    return fig


def sales_heatmap_fig_new(df: pd.DataFrame) -> go.Figure | None:
    if df.empty:
        return None
    z = _ensure_dates(df)
    if "event_date" not in z.columns or "sale_date" not in z.columns:
        return None
    z["days_out"] = (z["event_date"].dt.normalize() - z["sale_date"].dt.normalize()).dt.days
    z = z[(z["days_out"] >= 0) & (z["days_out"] <= 120)].copy()
    if z.empty:
        return None
    labels = ["0–7","8–14","15–30","31–60","61–90","91–120"]
    z["lead_bucket"] = pd.cut(z["days_out"], bins=[-1,7,14,30,60,90,120], labels=labels)
    z["weekday"] = z["sale_date"].dt.day_name()
    total = z["qty_sold"].sum()
    if total <= 0 or len(z["event_name"].unique()) < 20 or len(z) < 500:
        return None
    p = (
        z.groupby(["weekday", "lead_bucket"])["qty_sold"]
        .sum()
        .reset_index()
    )
    p["share"] = p["qty_sold"] / total
    wk = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    p["weekday"] = pd.Categorical(p["weekday"], wk, ordered=True)
    matrix = p.pivot(index="weekday", columns="lead_bucket", values="share").reindex(wk)
    fig = px.imshow(
        matrix,
        aspect="auto",
        color_continuous_scale="Blues",
        title=None,
    )
    fig.update_layout(
        template="plotly_white",
        title="",
        coloraxis_colorbar_title="Share of sales",
        coloraxis_colorbar_tickformat=".0%",
        margin=dict(l=10, r=10, t=20, b=10),
    )
    fig.update_traces(
        hovertemplate="Weekday: %{y}<br>Lead bucket: %{x}<br>Share: %{z:.1%}<extra></extra>",
        texttemplate="",
    )
    fig.update_yaxes(title="Weekday")
    fig.update_xaxes(title="Lead bucket (days before event)")
    return fig


def get_curve_sample_size(global_curve: pd.DataFrame) -> int:
    """Get the sample size (n) at day 0 from the pacing curve."""
    if global_curve.empty or "n" not in global_curve.columns:
        return 0
    # Day 0 is the last row (or find it explicitly)
    day_0 = global_curve[global_curve["d_bin"] == 0]
    if not day_0.empty:
        return int(day_0["n"].iloc[0])
    return int(global_curve["n"].iloc[-1]) if len(global_curve) > 0 else 0


# ---------------------------------------------------------------------------
# Avg tickets per event chart (Pre vs Post COVID) - force both periods visible
# ---------------------------------------------------------------------------
COVID_SPLIT = pd.Timestamp("2021-07-01")
TOP_K_TYPES = 12  # top event types by combined volume


def _periodize(df: pd.DataFrame) -> pd.DataFrame:
    d = df.dropna(subset=["sale_date", "event_date", "qty_sold"]).copy()
    d["period"] = np.where(d["sale_date"] < COVID_SPLIT, "Pre-COVID", "Post-COVID")
    return d


def _label_event_type(df: pd.DataFrame) -> pd.DataFrame:
    """Prefer event_type; fallback to event_part; else leave as 'Other'."""
    z = df.copy()
    if "event_type" in z.columns and z["event_type"].notna().any():
        z["_etype"] = z["event_type"].fillna("Other")
    elif "event_part" in z.columns:
        z["_etype"] = z["event_part"].fillna("Other")
    else:
        z["_etype"] = "Other"
    return z


def avg_tickets_per_event_by_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns tidy df with event_type, period, avg_tickets_per_event, n_events, total_tickets.
    Ensures both Pre-COVID and Post-COVID rows exist per type (filled with zeros if missing).
    """
    if df.empty:
        return pd.DataFrame()

    d = _label_event_type(_periodize(df))

    # One event instance = unique (event_name, event_date)
    ev = (
        d.groupby(["_etype", "period", "event_name", "event_date"], as_index=False)["qty_sold"]
        .sum()
        .rename(columns={"qty_sold": "tickets_event"})
    )

    g = (
        ev.groupby(["_etype", "period"], as_index=False)
        .agg(
            n_events=("event_name", "nunique"),
            total_tickets=("tickets_event", "sum"),
        )
    )

    # Ensure both periods exist per type
    all_types = g["_etype"].unique().tolist()
    full = (
        pd.MultiIndex.from_product([all_types, ["Pre-COVID", "Post-COVID"]], names=["_etype", "period"])
        .to_frame(index=False)
        .merge(g, on=["_etype", "period"], how="left")
        .fillna({"n_events": 0, "total_tickets": 0})
    )
    full["avg_tickets_per_event"] = full.apply(
        lambda row: row["total_tickets"] / row["n_events"] if row["n_events"] > 0 else 0, axis=1
    )
    return full


def pick_top_types(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    totals = (
        df.groupby("_etype", as_index=False)["total_tickets"]
        .sum()
        .sort_values("total_tickets", ascending=False)
        .head(TOP_K_TYPES)
    )
    keep = totals["_etype"].tolist()
    return df[df["_etype"].isin(keep)].copy()


def avg_tickets_fig(df: pd.DataFrame) -> tuple[go.Figure | None, bool]:
    """Return figure and flag indicating if pre period is missing."""
    by_type = avg_tickets_per_event_by_type(df)
    if by_type.empty:
        return None, False

    by_type = pick_top_types(by_type)
    by_type = by_type.sort_values(["_etype", "period"])

    # labels and category order
    by_type["_etype_label"] = by_type["_etype"]
    by_type["bar_label"] = "n=" + by_type["n_events"].astype(int).astype(str)

    fig = px.bar(
        by_type,
        x="_etype_label",
        y="avg_tickets_per_event",
        color="period",
        barmode="group",
        text="bar_label",
        labels={"_etype_label": "Event type", "avg_tickets_per_event": "Avg tickets per event"},
        category_orders={"period": ["Pre-COVID", "Post-COVID"]},
        color_discrete_map={"Pre-COVID": "#636EFA", "Post-COVID": "#EF553B"},
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(
        template="plotly_white",
        legend_title=None,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig.update_xaxes(tickangle=-25)

    # Detect if pre has no events in top types
    pre_missing = (by_type.query("period == 'Pre-COVID'")["n_events"] > 0).sum() == 0
    if pre_missing:
        fig.update_layout(title="Only Post-COVID events are visible under current filters. Broaden seasons/date range to see Pre-COVID.")

    return fig, pre_missing


# ---------------------------------------------------------------------------
# Category share shift chart
# ---------------------------------------------------------------------------
def share_shift_fig(pre: pd.DataFrame, post: pd.DataFrame) -> go.Figure | None:
    if pre.empty and post.empty:
        return None

    pre = pre.assign(window="Pre")
    post = post.assign(window="Post")
    combined = pd.concat([pre, post], ignore_index=True)

    if combined.empty:
        return None

    # Normalize to 100% per window
    combined["share_pct"] = combined.groupby("window")["share_ratio"].transform(
        lambda s: (s / s.sum()) * 100 if s.sum() > 0 else 0
    )

    # Filter out "Other (Unmapped)" unless >= 5%
    filtered = combined[(combined["event_type"] != "Other (Unmapped)") | (combined["share_ratio"] >= 0.05)]
    if filtered.empty:
        filtered = combined

    fig = px.bar(
        filtered,
        x="window",
        y="share_pct",
        color="event_type",
        barmode="stack",
        title="Category share shift (pre vs post)",
        text_auto=".1f",
    )
    fig.update_layout(template="plotly_white", legend_title="Event type")
    fig.update_yaxes(title="Share of tickets", range=[0, 105], ticksuffix="%", tickformat=".0f")
    fig.update_xaxes(categoryorder="array", categoryarray=["Pre", "Post"])
    return fig


# ---------------------------------------------------------------------------
# Top events chart
# ---------------------------------------------------------------------------
def top_events_fig(events: pd.DataFrame) -> go.Figure | None:
    if events.empty:
        return None

    data = events.head(10).copy()
    fig = go.Figure()

    for row in data.itertuples():
        fig.add_trace(
            go.Scatter(
                x=["Pre", "Post"],
                y=[row.qty_pre, row.qty_post],
                mode="lines+markers",
                name=row.event_name,
                hovertemplate="%{text}<br>%{x}: %{y:,} tickets<extra></extra>",
                text=[row.event_name, row.event_name],
            )
        )

    fig.update_layout(
        title="Top events: pre vs post performance",
        template="plotly_white",
        xaxis_title="Window",
        yaxis_title="Tickets sold",
        yaxis_tickformat=",",
    )
    fig.update_xaxes(type="category", categoryorder="array", categoryarray=["Pre", "Post"])
    return fig


# ---------------------------------------------------------------------------
# Sales heatmap
# ---------------------------------------------------------------------------
def sales_heatmap_fig(heat_df: pd.DataFrame) -> go.Figure | None:
    if heat_df.empty:
        return None

    matrix = (
        heat_df.pivot(index="weekday", columns="lead_bucket", values="share")
        .reindex(index=WEEKDAYS, columns=LEAD_BUCKET_LABELS, fill_value=0)
    )

    fig = px.imshow(
        matrix,
        aspect="auto",
        color_continuous_scale="Blues",
        title=None,
    )
    fig.update_layout(
        template="plotly_white",
        coloraxis_colorbar_title="% of weekday tickets",
        coloraxis_colorbar_tickformat=".0%",
        title="",
    )
    # Numbers on hover only
    fig.update_traces(
        hovertemplate="Weekday: %{y}<br>Lead bucket: %{x}<br>Share: %{z:.1%}<extra></extra>",
        texttemplate="",
    )
    fig.update_yaxes(title="Weekday")
    fig.update_xaxes(title="Lead bucket (days before event)")
    return fig


# ---------------------------------------------------------------------------
# Checkpoint cards (D-30, D-7)
# ---------------------------------------------------------------------------
def checkpoint_summary(table: pd.DataFrame, target_day: int, tolerance: int = 5) -> dict | None:
    """Compute checkpoint summary for D-30 or D-7 from watchlist."""
    if table.empty or "days_out" not in table.columns:
        return None

    matches = table[table["days_out"].between(target_day - tolerance, target_day + tolerance)]
    if matches.empty:
        return None

    if "tickets_so_far" in matches:
        weights = matches["tickets_so_far"].clip(lower=1)
    else:
        weights = np.ones(len(matches))
    actual = np.average(matches["sold_so_far_pct"], weights=weights)
    median = np.average(matches["typical_at_day_pct"], weights=weights)
    gap = actual - median

    return {
        "label": f"D-{target_day}",
        "actual": actual,
        "median": median,
        "gap": gap,
        "status": pacing.pace_status(gap),
    }


def render_checkpoint_cards(table: pd.DataFrame) -> None:
    st.subheader("Early / Late Checkpoints")

    summaries = [checkpoint_summary(table, 30), checkpoint_summary(table, 7)]
    valid = [s for s in summaries if s is not None]

    if not valid:
        st.info("Not enough upcoming events to compute early/late checkpoints.")
        return

    cols = st.columns(len(valid))
    for col, summary in zip(cols, valid):
        status = summary["status"]

        badge_color = "🟢" if status == "Ahead" else "🟡" if status == "On pace" else "🔴"

        col.metric(
            f"{summary['label']} cumulative %",
            _format_percent(summary["actual"]),
            f"{summary['gap']:+.1f} pp vs typical",
        )
        col.caption(f"{badge_color} Status: {summary['status']} ({summary['gap']:+.1f} percentage points)")


# ---------------------------------------------------------------------------
# Plot rendering helper
# ---------------------------------------------------------------------------
def render_plot(title: str, fig: go.Figure | None, key: str, subtitle: str | None = None) -> None:
    st.subheader(title)
    if subtitle:
        st.caption(subtitle)

    if fig is None or not fig.data:
        st.info("Not enough data to render this visual.")
        return

    st.plotly_chart(fig, use_container_width=True)

    try:
        ASSETS_DIR.mkdir(parents=True, exist_ok=True)
        png_path = ASSETS_DIR / f"{key}.png"
        fig.write_image(str(png_path), format="png", engine="kaleido", scale=2)
        with open(png_path, "rb") as handle:
            png_bytes = handle.read()
        st.download_button(
            "Download PNG",
            data=png_bytes,
            file_name=f"{key}.png",
            mime="image/png",
            key=f"{key}_png",
        )
    except Exception as exc:
        st.warning(f"PNG export unavailable: {exc}")


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="Krannert Dashboard", layout="wide", initial_sidebar_state="expanded")
    inject_styles()

    # Hero section (top of page)
    render_hero(DEFAULT_DATA_PATH)

    # Data source - clearer UI for preloaded data + optional updates
    st.sidebar.header("📊 Data Source")
    
    # Check if preloaded data exists
    has_preloaded = DEFAULT_DATA_PATH.exists()
    
    if has_preloaded:
        st.sidebar.success("✅ **Data preloaded** — Ready to explore!")
        st.sidebar.caption("Using: `sales_2016_2026_combined.csv`")
        
        # Expander for optional update
        with st.sidebar.expander("🔄 Upload updated data (optional)"):
            st.markdown("Upload a new CSV to replace the current dataset. "
                       "The file should have the same columns.")
            uploaded_sidebar = st.file_uploader("Drop new CSV here", type=["csv"], key="csv_upload")
    else:
        st.sidebar.warning("⚠️ No data file found")
        st.sidebar.markdown("Upload your sales CSV to get started:")
        uploaded_sidebar = st.sidebar.file_uploader("Upload CSV", type=["csv"], key="csv_upload")
    
    # Hidden local path input (for advanced users)
    local_path = str(DEFAULT_DATA_PATH)

    uploaded = uploaded_sidebar

    df, source_label = load_dataset(uploaded, local_path)
    if df is None:
        has_local = any(DATA_DIR.glob("*.csv"))
        if uploaded is None and not has_local:
            st.info("📋 No CSV detected — showing synthetic sample data for preview.")
            df = data_prep.make_fake_data()
            source_label = "Synthetic sample data"
        else:
            st.info("📁 Provide a CSV via upload to explore performance.")
            return

    # -------------------------------------------------------------------------
    # As-of date control (time-travel for pacing)
    # -------------------------------------------------------------------------
    # Default to last sale date in the data (safer for historical exports)
    if "sale_date" in df.columns and df["sale_date"].notna().any():
        default_asof = df["sale_date"].max().normalize()
    else:
        default_asof = pd.Timestamp.today().normalize()
    
    # Compute data date range for display
    sale_min = df["sale_date"].min() if "sale_date" in df.columns else None
    sale_max = df["sale_date"].max() if "sale_date" in df.columns else None
    
    st.sidebar.divider()
    st.sidebar.subheader("📅 As-of date")
    asof_date = st.sidebar.date_input(
        "Evaluate pacing as of",
        value=default_asof.date(),
        help="Set the 'current' date for pacing analysis. Useful to time-travel with historical data. Last updated sale date on merged csv: 10/29/2025",
    )
    asof_ts = pd.Timestamp(asof_date).normalize()
    
    # Data stamp
    data_stamp = f"**Data source:** {source_label}"
    if sale_min and sale_max:
        data_stamp += f" • Sales from {sale_min:%b %d, %Y} to {sale_max:%b %d, %Y}"
    data_stamp += f" • **As-of:** {asof_ts:%b %d, %Y}"
    data_stamp += f" • Rows: {len(df):,}"
    st.caption(data_stamp)

    # Filters
    filtered_df, filters, filters_summary = sidebar_filters(df)
    if filtered_df.empty:
        st.warning("No rows match the current filters. Try broadening your selection.")
        return

    # Derive aggregates
    aggregates = data_prep.derive_core(filtered_df)
    base = aggregates["base"]

    # Build watchlist using new pacing module with the selected as-of date
    # This computes Cum% as tickets_so_far / median_final_of_cohort (not event's own total)
    watch_table, watch_summary, fallback_tiers = pacing.build_watchlist(filtered_df, today=asof_ts)

    # KPIs (use as-of date for time-boxed calculations)
    kpis = compute_kpis(base, watch_summary, asof_ts)

    # KPI row
    render_kpis(kpis)

    # Download filtered CSV (ghost style)
    csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
    col_dl, col_note = st.columns([1, 2])
    with col_dl:
        st.markdown('<div class="ghost-btn">', unsafe_allow_html=True)
        st.download_button(
            "Download snapshot",
            data=csv_bytes,
            file_name="krannert_filtered.csv",
            mime="text/csv",
            key="filtered_csv",
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with col_note:
        st.markdown('<div class="muted">Exports respect your current filters and as-of date.</div>', unsafe_allow_html=True)

    # --- 1) Event Pacing Watchlist (move back up) ---
    st.subheader(f"1) 📊 Event Pacing Watchlist (as of {asof_ts:%B %d, %Y})")
    st.caption(
        "Statuses reflect the current filters on upcoming events; change filters to focus the watchlist. "
        "Baseline pacing chart uses a global historical curve for stability."
    )
    render_watchlist(watch_table, fallback_tiers, filters_summary)

    # --- 2) Booking Window Pacing ---
    st.subheader("2) Booking Window Pacing")
    with st.expander("How to read"):
        st.markdown(
            "Typical booking curve across your current filters. If your selection sits below the median at a given day-out, you’re behind pace.\n\n"
            "- X-axis: days before the event (counts down 120 → 0)\n"
            "- Y-axis: % of a typical final audience already sold\n"
            "- Blue line = typical pace; shaded = typical range\n"
            "- If your selection sits below the line at a given day-out, you’re behind pace."
        )

    mode = "Global"
    baseline_curve, cohort_label, cohort_n = build_baseline_curve(df, asof_ts, filters, mode)
    current_curve = build_current_curve(filtered_df, asof_ts)

    if baseline_curve is None or baseline_curve.empty:
        st.info("Not enough data to render the pacing curve.")
    else:
        plotly_config = {"displaylogo": False, "modeBarButtonsToRemove": ["toImage"]}
        pacing_fig = booking_window_fig(baseline_curve, current_curve)
        pacing_fig.update_layout(paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF")
        st.plotly_chart(pacing_fig, use_container_width=True, config=plotly_config)
        st.caption(
            f"Baseline: **Global** (n = {cohort_n:,} past events, stable across filters). "
            "Dashed line = current filtered upcoming events (shown only if ≥3 upcoming events)."
        )

    # --- 3) Sales Timing Heatmap (directly under booking) ---
    st.subheader("3) Sales Timing Heatmap")
    st.caption("Share of weekly sales by weekday × lead bucket. Darker = more of your sales tend to happen there.")
    no_zoom_config = {
        "displaylogo": False,
        "modeBarButtonsToRemove": [
            "zoom2d",
            "pan2d",
            "select2d",
            "lasso2d",
            "zoomIn2d",
            "zoomOut2d",
            "autoScale2d",
            "resetScale2d",
        ],
    }

    fig_heat = sales_heatmap_fig_new(filtered_df)
    if fig_heat is None:
        st.info("Not enough data to show pattern. Try widening seasons or channels.")
    else:
        fig_heat.update_layout(paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF")
        st.plotly_chart(fig_heat, use_container_width=True, config=no_zoom_config)

    # --- 4) Event Categories: Pre vs Post COVID ---
    st.subheader("4) Event Categories: Pre vs Post COVID")
    st.caption("Average audience size per category before vs after COVID (based on sale date). Pre = Sales Before 03/01/2020; Post = Sales After 07/01/2021.")
    with st.expander("How to read"):
        st.markdown(
            "- Compare category strength before and after COVID.\n"
            "- Bars show attendance; use to spot recovering or declining categories."
        )
    plotly_config = no_zoom_config
    cat_pre = fig_categories_pre(df)
    cat_post = fig_categories_post(df)
    cat_compare = fig_top_categories_pre_post(df, top_n=6)
    cols_cat = st.columns(2)
    with cols_cat[0]:
        if cat_pre is None:
            st.info("No pre-COVID data to render.")
        else:
            cat_pre.update_layout(paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF")
            st.plotly_chart(cat_pre, use_container_width=True, config=plotly_config)
    with cols_cat[1]:
        if cat_post is None:
            st.info("No post-COVID data to render.")
        else:
            cat_post.update_layout(paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF")
            st.plotly_chart(cat_post, use_container_width=True, config=plotly_config)

    st.markdown("**Top categories Pre vs Post COVID (Top 6)**")
    if cat_compare is None:
        st.info("No data for category comparison.")
    else:
        cat_compare.update_layout(paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF")
        st.plotly_chart(cat_compare, use_container_width=True, config=plotly_config)

    # --- 5) Top Events: Pre vs Post COVID ---
    st.subheader("5) Top Events: Pre vs Post COVID")
    st.caption("Shared events present in both periods, showing tickets pre vs post COVID.")
    ev_compare = fig_top_events_pre_post(df, k=12)
    if ev_compare is None:
        st.info("No events found in both periods.")
    else:
        ev_compare.update_layout(paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF")
        st.plotly_chart(ev_compare, use_container_width=True, config=plotly_config)


if __name__ == "__main__":
    main()