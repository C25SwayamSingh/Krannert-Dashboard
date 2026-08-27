"""
Krannert Dashboard — Streamlit application
"""
from __future__ import annotations

import re
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
# Monochrome design tokens (Figma-style: white surface, hairline borders, near-black ink)
INK = "#171717"        # primary ink & primary data series
INK_SOFT = "#525252"   # secondary text
INK_MUTED = "#8A8A8A"  # captions, hints
GRAY_SERIES = "#8C8C8C"  # secondary data series (validated ≥3:1 on white)
BORDER = "#E6E6E6"     # hairline borders
BORDER_STRONG = "#D9D9D9"
SURFACE = "#FFFFFF"
SURFACE_ALT = "#FAFAFA"
GRID = "#EFEFEF"       # chart gridlines
BAND = "rgba(23,23,23,0.06)"  # IQR band fill
RED = "#B42318"        # reserved status accent: "Behind" only
RED_WASH = "#FBEFEE"   # soft row wash for Behind rows
FONT_STACK = "'Inter', -apple-system, 'Segoe UI', 'Helvetica Neue', Arial, sans-serif"


def inject_styles() -> None:
    """Inject global CSS for the monochrome, Figma-like theme."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        html, body, .stApp, [class*="css"] {
            font-family: 'Inter', -apple-system, 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
            -webkit-font-smoothing: antialiased;
        }
        /* Global layout */
        .block-container {
            max-width: 1200px;
            padding-top: 1.25rem;
            padding-bottom: 4rem;
        }
        body { background: #FFFFFF; color: #171717; }
        .stApp { background: #FFFFFF; }

        /* Typography: tight, quiet, Figma-like */
        h1, h2, h3 { letter-spacing: -0.02em; font-weight: 600; color: #171717; }
        h3 { font-size: 1.15rem; margin-bottom: 0.25rem; }
        [data-testid="stCaptionContainer"], .stCaption { color: #8A8A8A; }
        p, li { color: #303030; }

        /* Sidebar: light panel with a hairline divider */
        [data-testid="stSidebar"] {
            background: #FAFAFA;
            border-right: 1px solid #E6E6E6;
        }
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            color: #8A8A8A;
            font-weight: 600;
        }
        [data-testid="stSidebar"] hr { border-color: #E6E6E6; margin: 0.75rem 0; }
        [data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 0.6rem; }

        /* Brand mark at the top of the sidebar */
        .brand {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 2px 4px 14px;
        }
        .brand-mark {
            width: 30px;
            height: 30px;
            border-radius: 8px;
            background: #171717;
            color: #FFFFFF;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 600;
            font-size: 14px;
            letter-spacing: -0.02em;
            flex: none;
        }
        .brand-text { line-height: 1.15; }
        .brand-name {
            font-weight: 600;
            font-size: 14px;
            color: #171717;
            letter-spacing: -0.01em;
        }
        .brand-sub { font-size: 11px; color: #8A8A8A; }

        /* Section navigation — icon + label rows, like a standard
           dashboard's left rail */
        .nav { display: flex; flex-direction: column; gap: 2px; margin-bottom: 4px; }
        .nav-item {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 7px 10px;
            border-radius: 6px;
            color: #303030 !important;
            text-decoration: none !important;
            font-size: 13.5px;
            font-weight: 500;
            line-height: 1.3;
            transition: background 0.1s ease, color 0.1s ease;
        }
        .nav-item:hover { background: #EFEFEF; color: #171717 !important; }
        .nav-item svg { flex: none; width: 17px; height: 17px; stroke: #8A8A8A; }
        .nav-item:hover svg { stroke: #171717; }
        .nav-label { font-size: 11px; text-transform: uppercase; letter-spacing: 0.07em;
                     color: #8A8A8A; font-weight: 600; padding: 6px 10px 2px; }

        /* Page title — large and confident, like a standard dashboard header */
        .page-title {
            font-size: 1.7rem;
            font-weight: 700;
            letter-spacing: -0.03em;
            color: #171717;
            margin-bottom: 2px;
        }

        /* Cards */
        .card {
            background: #FFFFFF;
            border-radius: 8px;
            padding: 16px 20px 14px;
            border: 1px solid #E6E6E6;
            margin-bottom: 14px;
        }
        .card h3, .card h4, .card h5 { margin-top: 0.1rem; }

        /* Metric tiles: bordered white cards, hairline, no shadow */
        [data-testid="stMetric"] {
            background: #FFFFFF;
            border: 1px solid #E6E6E6;
            border-radius: 8px;
            padding: 12px 16px;
        }
        [data-testid="stMetricLabel"] {
            color: #8A8A8A;
            font-size: 12px;
        }
        [data-testid="stMetricValue"] {
            color: #171717;
            font-weight: 600;
            letter-spacing: -0.02em;
        }
        [data-testid="stMetricDelta"] {
            font-size: 12px;
            color: #525252 !important;
            background: #F5F5F5 !important;
        }
        [data-testid="stMetricDelta"] svg { fill: #525252 !important; color: #525252 !important; }

        /* Alerts (info/success/warning): quiet monochrome panels */
        [data-testid="stAlertContainer"] {
            background: #FAFAFA !important;
            border: 1px solid #E6E6E6 !important;
            border-radius: 8px !important;
            color: #303030 !important;
        }
        [data-testid="stAlertContainer"] p { color: #303030 !important; }
        [data-testid="stAlertContainer"] svg { fill: #525252 !important; }

        /* KPI tiles — one CSS grid, so all four cards share the exact
           same height no matter how their labels or notes wrap */
        .kpi-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 14px;
            align-items: stretch;
            margin: 4px 0 8px;
        }
        @media (max-width: 900px) {
            .kpi-grid { grid-template-columns: repeat(2, 1fr); }
        }
        .kpi-card { padding: 16px 18px 14px; }
        .kpi-tile {
            display: flex;
            flex-direction: column;
            margin-bottom: 0;
        }
        /* Label block reserves two lines so every number starts on the
           same baseline; the note is pinned to the bottom of the card. */
        .kpi-tile-label {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 6px;
            min-height: 2.6em;
        }
        .kpi-tile .kpi-value { line-height: 1.15; margin-bottom: 6px; }
        .kpi-tile .kpi-delta { margin-top: auto; }

        /* Info-icon tooltips on the headline numbers */
        .kpi-tooltip-wrap {
            position: relative;
            display: inline-flex;
            align-items: center;
            flex: none;
        }
        .kpi-tooltip-icon {
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 15px;
            height: 15px;
            border-radius: 50%;
            background: #EDEDED;
            color: #525252;
            font-size: 10px;
            font-weight: 600;
            flex: none;
        }
        .kpi-tooltip-icon:hover + .kpi-tooltip-text,
        .kpi-tooltip-text:hover { visibility: visible; opacity: 1; }
        .kpi-tooltip-text {
            visibility: hidden;
            opacity: 0;
            position: absolute;
            bottom: 100%;
            margin-bottom: 6px;
            background: #171717;
            color: #fff;
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 400;
            letter-spacing: 0;
            text-transform: none;
            width: max-content;
            max-width: min(280px, 40vw);
            white-space: normal;
            overflow-wrap: break-word;
            z-index: 1000;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
            transition: opacity 0.1s ease-in-out;
            line-height: 1.45;
        }
        .kpi-label {
            font-size: 11px;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: #8A8A8A;
            margin-bottom: 2px;
        }
        .kpi-value {
            font-size: 26px;
            font-weight: 600;
            letter-spacing: -0.02em;
            margin-bottom: 2px;
            color: #171717;
        }
        .kpi-delta { font-size: 12px; color: #525252; }
        .kpi-help {
            cursor: help;
            font-size: 0.9rem;
            display: inline-block;
            padding-left: 6px;
            line-height: 1;
            opacity: 0.85;
        }

        /* Buttons: solid black primary, quiet gray ghost, small radius */
        .stButton>button, .stDownloadButton>button {
            border-radius: 6px !important;
            border: 1px solid #D9D9D9 !important;
            background: #FFFFFF !important;
            color: #171717 !important;
            font-weight: 500 !important;
            box-shadow: none !important;
            transition: border-color 0.1s ease, background 0.1s ease;
        }
        .stButton>button:hover, .stDownloadButton>button:hover {
            border-color: #171717 !important;
            background: #FAFAFA !important;
        }
        .primary-btn button, .primary-btn>button {
            background: #171717 !important;
            color: #FFFFFF !important;
            border: 1px solid #171717 !important;
        }
        .ghost-btn button, .ghost-btn>button {
            background: transparent !important;
            color: #171717 !important;
            border: 1px solid #D9D9D9 !important;
        }

        /* Badges & helper text */
        .pill {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            background: #FAFAFA;
            border: 1px solid #E6E6E6;
            color: #525252;
            font-size: 11px;
        }
        .muted { color: #8A8A8A; font-size: 13px; }

        /* Inputs: hairline borders */
        [data-testid="stSidebar"] [data-baseweb="select"] > div,
        [data-testid="stSidebar"] input {
            border-color: #E6E6E6 !important;
            background: #FFFFFF;
        }

        /* Hide Streamlit settings (theme chooser) to keep a single enforced theme */
        div[data-testid="stToolbar"] button[title="Settings"] { display: none !important; }

        /* Compact, readable help list */
        .help-list {
            margin: 0.25rem 0 0.5rem 0;
            padding-left: 1.0rem;
            line-height: 1.6;
            color: #303030;
            font-size: 14px;
        }
        .help-list li { margin: 0.2rem 0; }

        /* Expanders: bordered, flat */
        [data-testid="stExpander"] {
            border: 1px solid #E6E6E6 !important;
            border-radius: 8px !important;
            background: #FFFFFF;
        }
        [data-testid="stExpander"] summary { color: #525252; }
        [data-testid="stExpander"] .streamlit-expanderContent {
            background: #FFFFFF;
            border-radius: 8px;
            padding: 0.5rem 0.75rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# Section anchors, shared between the sidebar nav and the page headings so
# the two can never drift apart.
SECTIONS = [
    ("overview", "Overview", "M3 3h7v7H3zM14 3h7v7h-7zM14 14h7v7h-7zM3 14h7v7H3z"),
    ("watchlist", "Needs attention", "M12 9v4M12 17h.01M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"),
    ("booking", "Booking pace", "M3 3v18h18M7 15l4-5 4 3 5-7"),
    ("timing", "When people book", "M8 2v4M16 2v4M3 10h18M5 4h14a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2z"),
    ("categories", "Categories", "M12 20V10M18 20V4M6 20v-4"),
    ("returning", "Returning shows", "M3 12a9 9 0 0 1 15-6.7L21 8M21 3v5h-5M21 12a9 9 0 0 1-15 6.7L3 16M3 21v-5h5"),
]


def render_sidebar_nav() -> None:
    """Brand mark plus jump links to each section of the page."""
    st.sidebar.markdown(
        '<div class="brand">'
        '<div class="brand-mark">K</div>'
        '<div class="brand-text">'
        '<div class="brand-name">Krannert Center</div>'
        '<div class="brand-sub">Ticket sales</div>'
        "</div></div>",
        unsafe_allow_html=True,
    )
    items = "".join(
        f'<a class="nav-item" href="#{anchor}">'
        f'<svg viewBox="0 0 24 24" fill="none" stroke-width="1.8" '
        f'stroke-linecap="round" stroke-linejoin="round"><path d="{path}"/></svg>'
        f"<span>{label}</span></a>"
        for anchor, label, path in SECTIONS
    )
    st.sidebar.markdown(
        f'<div class="nav-label">Sections</div><nav class="nav">{items}</nav>',
        unsafe_allow_html=True,
    )
    st.sidebar.divider()


def style_mono(fig: go.Figure) -> go.Figure:
    """Apply the monochrome chart style: Inter, white surface, recessive grid/axes."""
    fig.update_layout(
        template="plotly_white",
        font=dict(family=FONT_STACK, color=INK, size=12),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        colorway=[INK, GRAY_SERIES, "#BFBFBF", "#5E5E5E"],
        legend=dict(font=dict(color=INK_SOFT, size=11)),
        hoverlabel=dict(
            bgcolor=INK,
            font=dict(family=FONT_STACK, color="#FFFFFF", size=12),
            bordercolor=INK,
        ),
    )
    fig.update_xaxes(
        gridcolor=GRID, linecolor=BORDER_STRONG, tickcolor=BORDER_STRONG,
        title_font=dict(color=INK_MUTED, size=11), tickfont=dict(color=INK_SOFT, size=11),
        zerolinecolor=GRID,
    )
    fig.update_yaxes(
        gridcolor=GRID, linecolor=BORDER_STRONG, tickcolor=BORDER_STRONG,
        title_font=dict(color=INK_MUTED, size=11), tickfont=dict(color=INK_SOFT, size=11),
        zerolinecolor=GRID,
    )
    return fig


# Raw exports prefix the performance with a weekday abbreviation ("Th/Hamlet")
# and suffix the price code ("Hamlet – UI"). The dashboard shows the weekday
# and price category in their own words, so strip both from display names.
_WEEKDAY_PREFIX = re.compile(r"^(Mo|Tu|We|Th|Fr|Sa|Su)/", re.IGNORECASE)


def prettify_event(label: str) -> str:
    """'Th/The Play That Goes Wrong – UI' -> 'The Play That Goes Wrong (UI tickets)'."""
    if not isinstance(label, str):
        return str(label)
    text = _WEEKDAY_PREFIX.sub("", label.strip())
    if " – " in text:
        name, _, code = text.rpartition(" – ")
        code = code.strip()
        if name and code:
            return f"{name.strip()} ({code} tickets)"
    return text


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
    st.markdown('<span id="overview"></span>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">Krannert Sales Insights</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="muted">See which upcoming shows are selling behind pace, '
        "when audiences book, and how categories have recovered since COVID.</div>",
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

    all_seasons = filters.get("seasons") or []
    parts = []
    if all_seasons:
        parts.append(f"All {len(all_seasons)} seasons ({all_seasons[0]} to {all_seasons[-1]})")

    channels = filters.get("channels") or []
    if channels:
        parts.append(summarize(channels, "Sold through"))

    start, end = filters.get("date_range", (None, None))
    if start is not None and end is not None:
        parts.append(f"shows between {start.date():%b %d, %Y} and {end.date():%b %d, %Y}")

    return "Showing: " + "; ".join(parts) if parts else ""


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
    st.sidebar.subheader("What is shown")

    # Seasons are shown, not filtered. Every season in the file is always
    # included: the historical seasons are what the pacing benchmarks are
    # built from, so letting users drop them silently weakened the baseline
    # while the headline metrics appeared unchanged.
    seasons = defaults["all_seasons"]
    selected_seasons = seasons
    if seasons:
        st.sidebar.markdown("**Seasons included**")
        st.sidebar.caption(
            f"All {len(seasons)} seasons in the file, {seasons[0]} through {seasons[-1]}. "
            "Past seasons are what upcoming shows get compared against, so they are always included. "
            "To narrow the view, use the show date range below."
        )

    channels = sorted(df["channel"].dropna().unique()) if "channel" in df.columns else []
    selected_channels = st.sidebar.multiselect(
        "Sold through", channels, default=channels, placeholder="All sales channels"
    )

    min_date, max_date = defaults["date_bounds"]
    date_range = st.sidebar.date_input(
        "Show date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        help="Limits the view to performances happening between these two dates.",
    )
    if isinstance(date_range, tuple):
        start_date, end_date = date_range
    else:
        start_date = end_date = date_range

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    mask = pd.Series(True, index=df.index)
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


def _kpi_card_html(
    label: str,
    value: str,
    note: str,
    help_text: str,
    accent: str = INK,
    align: str = "left",
) -> str:
    """One headline number as a self-contained card (HTML fragment)."""
    safe_help = help_text.replace('"', "&quot;")
    anchor = "right: 0; left: auto;" if align == "right" else "left: 0; right: auto;"
    return (
        '<div class="card kpi-card kpi-tile">'
        f'<div class="kpi-label kpi-tile-label"><span>{label}</span>'
        '<span class="kpi-tooltip-wrap"><span class="kpi-tooltip-icon">i</span>'
        f'<span class="kpi-tooltip-text" style="{anchor}">{safe_help}</span></span></div>'
        f'<div class="kpi-value" style="color:{accent}">{value}</div>'
        f'<div class="kpi-delta">{note}</div>'
        "</div>"
    )


def render_kpis(kpis: dict) -> None:
    """
    All four headline numbers in ONE markdown block laid out with CSS grid,
    so the cards are equal-height by construction — Streamlit columns wrap
    each card in its own div stack and the heights drift apart.
    """
    trend = ""
    if kpis["delta_30d"] is not None:
        pct = kpis["delta_30d"] * 100
        direction = "more" if pct >= 0 else "fewer"
        trend = f"{abs(pct):.0f}% {direction} than the 30 days before"

    behind_note = (
        f"{kpis['behind_detail']} shows" if kpis["behind_detail"] != "–" else "nothing to measure"
    )

    cards = [
        _kpi_card_html(
            "Tickets sold recently",
            _format_int(kpis["tickets_30d"]),
            trend or "in the last 30 days",
            "Tickets sold in the last 30 days, compared with the 30 days before that.",
        ),
        _kpi_card_html(
            "Average ticket price",
            _format_currency(kpis["avg_price_30d"]),
            "over the last 30 days",
            "What the average ticket actually sold for over the last 30 days, "
            "including discounted and free tickets.",
        ),
        _kpi_card_html(
            "Shows coming up",
            _format_int(kpis["open_events"]),
            "in the next 30 days",
            "Performances happening in the next 30 days that are still selling tickets.",
            align="right",
        ),
        _kpi_card_html(
            "Selling behind pace",
            _format_percent(kpis["behind_pct"], decimals=0),
            behind_note,
            "Of the upcoming shows we can measure, the share selling slower than similar "
            "past shows were at the same point before their date. These are the ones worth "
            "a marketing or pricing look.",
            accent=RED if kpis["behind_pct"] > 0 else INK,
            align="right",
        ),
    ]
    st.markdown(f'<div class="kpi-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Watchlist table (using new pacing module)
# ---------------------------------------------------------------------------
def render_watchlist_summary(
    watchlist_df: pd.DataFrame,
    upcoming_window_days: int = 120,
    watch_summary: dict | None = None,
) -> None:
    # An empty table with evaluated events means every row is ≥98% sold —
    # fall through so the pills still report the true evaluated counts.
    has_summary = bool(watch_summary and watch_summary.get("evaluated"))
    if (watchlist_df is None or watchlist_df.empty) and not has_summary:
        st.caption(f"Showing **0** upcoming events (≤{upcoming_window_days}d) that match your filters.")
        return

    # Normalize the status labels robustly
    status_col = None
    for c in ["status", "Status", "STATUS"]:
        if watchlist_df is not None and c in watchlist_df.columns:
            status_col = c
            break

    if status_col is None and not has_summary:
        # No status column; just show the total
        st.caption(f"Showing **{len(watchlist_df)}** upcoming events (≤{upcoming_window_days}d) that match your filters.")
        return

    # Prefer the full evaluated counts from build_watchlist — the displayed
    # table is truncated to the most at-risk rows, so counting its statuses
    # would always look overwhelmingly "behind".
    if watch_summary and watch_summary.get("evaluated"):
        n_total = int(watch_summary["evaluated"])
        n_behind = int(watch_summary.get("behind", 0))
        n_on = int(watch_summary.get("on_pace", 0))
        n_ahead = int(watch_summary.get("ahead", 0))
    else:
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
          .pill-behind { background: #B42318; color: #FFFFFF; }
          .pill-on { background: #E8E8E8; color: #171717; }
          .pill-ahead { background: #FFFFFF; color: #525252; border: 1px solid #D9D9D9; }
        </style>
    """, unsafe_allow_html=True)

    st.markdown(
        (f"**{n_total} shows are coming up in the next {upcoming_window_days} days:** "
         f"<span class='pill-status pill-behind'>{n_behind} behind pace</span> "
         f"<span class='pill-status pill-on'>{n_on} on pace</span> "
         f"<span class='pill-status pill-ahead'>{n_ahead} ahead</span>"
        ),
        unsafe_allow_html=True
    )


def render_price_code_guide(df: pd.DataFrame) -> None:
    """
    Reference table for the box-office price codes that appear throughout the
    dashboard. We report what the data shows (volume, average price) rather
    than guessing at what each code stands for.
    """
    if "event_part" not in df.columns:
        return

    sold = df[(df["qty_sold"] > 0) & df["event_part"].notna()].copy()
    sold["event_part"] = sold["event_part"].astype(str).str.strip().str.upper()
    sold = sold[sold["event_part"] != ""]
    if sold.empty:
        return

    g = (
        sold.groupby("event_part")
        .agg(tickets=("qty_sold", "sum"), revenue=("amount", "sum"))
        .sort_values("tickets", ascending=False)
        .head(12)
    )
    g["avg"] = (g["revenue"] / g["tickets"]).fillna(0)
    total = sold["qty_sold"].sum()

    with st.expander("What do the ticket codes (UI, SA, SC…) mean?"):
        st.markdown(
            "These are your box-office price categories, carried straight through "
            "from the sales export. We do not rename them, so they match what your "
            "ticketing system shows. Here are the most common ones, with what they "
            "actually sold for:"
        )
        table = pd.DataFrame(
            {
                "Code": g.index,
                "Tickets sold": [f"{int(t):,}" for t in g["tickets"]],
                "Share of all tickets": [f"{t / total:.0%}" for t in g["tickets"]],
                "Average price": [f"${a:,.2f}" for a in g["avg"]],
            }
        )
        st.dataframe(table, hide_index=True)
        st.caption(
            "A $0.00 average means those tickets are comps or otherwise free. "
            "If you would like these shown by name instead of code, send us the "
            "list your box office uses and we will label them."
        )


# Plain-English column headings for the watchlist table
HEADER_LABELS = {
    "event": "<b>Show</b>",
    "days_out": "<b>Days until show</b>",
    "sold_so_far_pct": "<b>Sold so far</b>",
    "typical_at_day_pct": "<b>Usually sold by now</b>",
    "gap_pp": "<b>Difference</b>",
    "tickets_so_far": "<b>Tickets sold</b>",
    "tickets_at_risk": "<b>Tickets at risk</b>",
    "status": "<b>Status</b>",
    "cohort": "<b>Compared with</b>",
}

# Relative column widths — show titles and the comparison note need the room;
# the numeric columns are narrow, so they stop wrapping onto three lines.
COLUMN_WIDTHS = {
    "event": 2.6,
    "days_out": 1.0,
    "sold_so_far_pct": 0.95,
    "typical_at_day_pct": 1.15,
    "gap_pp": 0.95,
    "tickets_so_far": 0.95,
    "tickets_at_risk": 0.95,
    "status": 0.85,
    "cohort": 2.0,
}


def render_watchlist(
    watch_table: pd.DataFrame,
    fallback_tiers: set,
    filters_summary: str,
    watch_summary: dict | None = None,
) -> None:
    """Render the event pacing watchlist with Plotly table for visibility."""
    with st.expander("How to read this table"):
        st.markdown("""
Each row is one upcoming show. We compare how many tickets it has sold so far
against how many similar shows had sold at the same point before their date.

- **Days until show** — how long until the performance.
- **Sold so far** — tickets sold, as a share of what a similar show usually sells in total.
- **Usually sold by now** — where similar past shows stood at this same point.
- **Difference** — how far ahead or behind the show is. A difference of −20 means
  it has sold 20% less of its expected audience than usual by this point.
- **Tickets sold** — the actual ticket count so far.
- **At risk** — roughly how many tickets the show is short, if the gap does not close.
- **Status** — "Behind" if it is more than 5 points under the usual pace,
  "Ahead" if more than 5 points over, "On pace" in between.
- **Compared with** — the group of past shows used as the yardstick.

Codes like UI, SA and SC are your own box-office price categories. The same
performance appears once per category, because each one sells at its own pace.
        """)

    # Render concise summary
    try:
        window_days = pacing.D_MAX
    except AttributeError:
        window_days = 120
    render_watchlist_summary(watch_table, upcoming_window_days=window_days, watch_summary=watch_summary)

    if watch_table.empty:
        st.info("No shows are coming up in the next 120 days, or there is not enough past data to compare against.")
        return

    if watch_summary and watch_summary.get("evaluated", 0) > len(watch_table):
        st.caption(
            f"The table below lists the {len(watch_table)} shows most at risk, "
            f"out of {watch_summary['evaluated']} we can measure. "
            "Shows closest to selling out are left off."
        )

    # Column order and configs
    display_cols = [
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
    display = watch_table[[c for c in display_cols if c in watch_table.columns]].copy()
    display_cols = [c for c in display_cols if c in display.columns]

    # Behind rows get a soft red wash. Red is reserved for Behind, and the
    # status word itself carries the meaning, so color is never the only cue.
    def status_color(status: str) -> str:
        if status == "Behind":
            return RED_WASH
        return "#FFFFFF"

    # Values read as plain numbers: whole percents, signed differences, no
    # decimals to squint at.
    fmt = {
        "event": prettify_event,
        "days_out": lambda v: "Today" if v == 0 else ("1 day" if v == 1 else f"{v:d} days"),
        "sold_so_far_pct": lambda v: f"{v:.0f}%",
        "typical_at_day_pct": lambda v: f"{v:.0f}%",
        "gap_pp": lambda v: f"{v:+.0f}",
        "tickets_so_far": lambda v: f"{int(v):,}",
        "tickets_at_risk": lambda v: "—" if v <= 0 else f"{int(round(v)):,}",
    }
    formatted_cols = []
    for col in display_cols:
        if col in fmt:
            formatted_cols.append(display[col].apply(fmt[col]).tolist())
        else:
            formatted_cols.append(display[col].tolist())

    # Row-wise colors; the Status column's text goes red on Behind rows
    row_colors = [[status_color(s) for s in display["status"]]] * len(display_cols)
    status_font = [RED if s == "Behind" else INK for s in display["status"]]
    font_colors = [
        status_font if col == "status" else [INK] * len(display)
        for col in display_cols
    ]

    fig = go.Figure(
        data=[
            go.Table(
                columnwidth=[COLUMN_WIDTHS.get(c, 1.0) for c in display_cols],
                header=dict(
                    values=[
                        HEADER_LABELS.get(c, c) for c in display_cols
                    ],
                    fill_color=SURFACE_ALT,
                    line_color=BORDER,
                    font=dict(family=FONT_STACK, color=INK_SOFT, size=12),
                    align="left",
                    height=32,
                ),
                cells=dict(
                    values=formatted_cols,
                    fill_color=row_colors,
                    line_color="#F0F0F0",
                    align="left",
                    font=dict(family=FONT_STACK, color=font_colors, size=12),
                    height=30,
                ),
            )
        ]
    )
    fig.update_layout(
        font=dict(family=FONT_STACK),
        paper_bgcolor=SURFACE,
        margin=dict(l=0, r=0, t=10, b=0),
        height=min(600, 35 + 30 * len(display) + 60),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

    # Summary stats with emojis — use full evaluated counts, not the truncated
    # most-at-risk table, which would always skew "behind"
    if watch_summary and watch_summary.get("evaluated"):
        behind_count = int(watch_summary.get("behind", 0))
        ahead_count = int(watch_summary.get("ahead", 0))
        on_pace_count = int(watch_summary.get("on_pace", 0))
    else:
        behind_count = int((watch_table["status"] == "Behind").sum())
        ahead_count = int((watch_table["status"] == "Ahead").sum())
        on_pace_count = int((watch_table["status"] == "On pace").sum())
    def status_tile(col, label: str, value: int, note: str, accent: str = INK) -> None:
        col.markdown(
            f'<div class="card kpi-card" style="margin-bottom:0">'
            f'<div class="kpi-label">{label}</div>'
            f'<div class="kpi-value" style="color:{accent}">{value:,}</div>'
            f'<div class="kpi-delta">{note}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )

    c1, c2, c3 = st.columns(3)
    status_tile(c1, "Behind pace", behind_count, "worth a closer look", accent=RED)
    status_tile(c2, "On pace", on_pace_count, "selling as expected")
    status_tile(c3, "Ahead", ahead_count, "selling faster than usual")


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
            fillcolor=BAND,
            name="Typical range for most shows",
            hoverinfo="skip",
        )
    )

    # Median line
    fig.add_trace(
        go.Scatter(
            x=global_curve["d_bin"],
            y=global_curve["median_pct"],
            mode="lines",
            line=dict(color=INK, width=2),
            name="Typical pace",
            hovertemplate="%{x} days before the show: %{y:.0f}% sold<extra></extra>",
        )
    )

    # Overlay current selection if provided
    if current_curve is not None and not current_curve.empty:
        fig.add_trace(
            go.Scatter(
                x=current_curve["d_bin"],
                y=current_curve["cum_pct"],
                mode="lines",
                line=dict(color=GRAY_SERIES, width=2, dash="dash"),
                name="Your upcoming shows",
                hovertemplate="%{x} days before the show: %{y:.0f}% sold<extra></extra>",
            )
        )

    # D-30 and D-7 checkpoint lines
    for d, label in [(30, "30 days out"), (7, "1 week out")]:
        fig.add_vline(
            x=d,
            line_dash="dot",
            line_color=BORDER_STRONG,
            annotation_text=label,
            annotation_position="top",
            annotation_font=dict(color=INK_MUTED, size=11),
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
        title_text="Share of the show's final ticket sales",
        range=[0, 105],
        ticksuffix="%",
        tickformat=".0f",
        rangemode="tozero",
    )

    # X axis inverted: 120 → 0
    fig.update_xaxes(
        title_text="Days before the show",
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
        z.groupby(["weekday", "lead_bucket"], observed=False)["qty_sold"]
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
        color_continuous_scale="Greys",
        title=None,
    )
    fig.update_layout(
        template="plotly_white",
        title="",
        coloraxis_colorbar_title="Share of<br>all sales",
        coloraxis_colorbar_tickformat=".0%",
        margin=dict(l=10, r=10, t=20, b=10),
    )
    fig.update_traces(
        hovertemplate="Bought on %{y}, %{x} days ahead<br>%{z:.1%} of all sales<extra></extra>",
        texttemplate="",
    )
    fig.update_yaxes(title="Day the ticket was bought")
    fig.update_xaxes(title="How far ahead the ticket was bought (days)")
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
    d["period"] = np.where(d["sale_date"] < COVID_SPLIT, "Before COVID", "After COVID")
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
        pd.MultiIndex.from_product([all_types, ["Before COVID", "After COVID"]], names=["_etype", "period"])
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
    by_type["bar_label"] = by_type["n_events"].astype(int).astype(str) + " shows"

    fig = px.bar(
        by_type,
        x="_etype_label",
        y="avg_tickets_per_event",
        color="period",
        barmode="group",
        text="bar_label",
        labels={"_etype_label": "Ticket code", "avg_tickets_per_event": "Average audience per show"},
        category_orders={"period": ["Before COVID", "After COVID"]},
        color_discrete_map={"Before COVID": GRAY_SERIES, "After COVID": INK},
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
        fig.update_layout(title="Only shows from after COVID appear in this view. Widen the show date range to include earlier ones.")

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
        color_continuous_scale="Greys",
        title=None,
    )
    fig.update_layout(
        template="plotly_white",
        coloraxis_colorbar_title="Share of<br>that day's sales",
        coloraxis_colorbar_tickformat=".0%",
        title="",
    )
    # Numbers on hover only
    fig.update_traces(
        hovertemplate="Bought on %{y}, %{x} days ahead<br>%{z:.1%} of all sales<extra></extra>",
        texttemplate="",
    )
    fig.update_yaxes(title="Day the ticket was bought")
    fig.update_xaxes(title="How far ahead the ticket was bought (days)")
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
        "label": f"{target_day} days",
        "actual": actual,
        "median": median,
        "gap": gap,
        "status": pacing.pace_status(gap),
    }


def render_checkpoint_cards(table: pd.DataFrame) -> None:
    st.subheader("Checkpoints before the show")

    summaries = [checkpoint_summary(table, 30), checkpoint_summary(table, 7)]
    valid = [s for s in summaries if s is not None]

    if not valid:
        st.info("Not enough upcoming shows to work out checkpoints.")
        return

    cols = st.columns(len(valid))
    for col, summary in zip(cols, valid):
        status = summary["status"]

        col.metric(
            f"{summary['label']} before the show",
            _format_percent(summary["actual"], decimals=0),
            f"{summary['gap']:+.0f} vs usual",
            delta_color="off",
        )
        col.caption(f"{status} — usually {summary['median']:.0f}% sold by this point")


# ---------------------------------------------------------------------------
# Plot rendering helper
# ---------------------------------------------------------------------------
def render_plot(title: str, fig: go.Figure | None, key: str, subtitle: str | None = None) -> None:
    st.subheader(title)
    if subtitle:
        st.caption(subtitle)

    if fig is None or not fig.data:
        st.info("Not enough data to show this chart.")
        return

    st.plotly_chart(fig, use_container_width=True)

    try:
        ASSETS_DIR.mkdir(parents=True, exist_ok=True)
        png_path = ASSETS_DIR / f"{key}.png"
        fig.write_image(str(png_path), format="png", engine="kaleido", scale=2)
        with open(png_path, "rb") as handle:
            png_bytes = handle.read()
        st.download_button(
            "Download this chart as an image",
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

    render_sidebar_nav()

    # Data source - clearer UI for preloaded data + optional updates
    st.sidebar.header("Data")
    
    # Check if preloaded data exists
    has_preloaded = DEFAULT_DATA_PATH.exists()
    
    if has_preloaded:
        st.sidebar.success("**Data preloaded** — ready to explore.")
        st.sidebar.caption("File: sales_2016_2026_combined.csv")
        
        # Expander for optional update
        with st.sidebar.expander("Use a newer file instead"):
            st.markdown("Upload a newer sales export to use instead. "
                       "It needs the same columns as the current file.")
            uploaded_sidebar = st.file_uploader("Choose a file", type=["csv"], key="csv_upload")
    else:
        st.sidebar.warning("No data file found")
        st.sidebar.markdown("Upload a sales export to get started:")
        uploaded_sidebar = st.sidebar.file_uploader("Choose a file", type=["csv"], key="csv_upload")
    
    # Hidden local path input (for advanced users)
    local_path = str(DEFAULT_DATA_PATH)

    uploaded = uploaded_sidebar

    df, source_label = load_dataset(uploaded, local_path)
    if df is None:
        has_local = any(DATA_DIR.glob("*.csv"))
        if uploaded is None and not has_local:
            st.info("No CSV detected — showing synthetic sample data for preview.")
            df = data_prep.make_fake_data()
            source_label = "Synthetic sample data"
        else:
            st.info("Provide a CSV via upload to explore performance.")
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
    st.sidebar.subheader("Reporting date")
    asof_date = st.sidebar.date_input(
        "Count ticket sales through",
        value=default_asof.date(),
        help="Treat this as 'today'. Sales after this date are ignored, so you can "
             "look back at how things stood on an earlier date.",
    )
    asof_ts = pd.Timestamp(asof_date).normalize()

    # Data stamp
    data_stamp = ""
    if sale_min and sale_max:
        data_stamp += f"Ticket sales from {sale_min:%B %Y} through {sale_max:%B %d, %Y}. "
    data_stamp += f"Counting sales through **{asof_ts:%B %d, %Y}**. "
    data_stamp += f"{len(df):,} sales records."
    st.caption(data_stamp)

    # Filters
    filtered_df, filters, filters_summary = sidebar_filters(df)
    if filtered_df.empty:
        st.warning("No sales match the current date range. Try widening it.")
        return

    # Derive aggregates
    aggregates = data_prep.derive_core(filtered_df)
    base = aggregates["base"]

    # Build watchlist using new pacing module with the selected as-of date
    # This computes Cum% as tickets_so_far / median_final_of_cohort (not event's own total)
    # Filters choose WHICH upcoming events are evaluated; the benchmark
    # cohorts always come from the full dataset so a narrow (e.g. future-only)
    # date range can't hollow out the historical baseline.
    watch_table, watch_summary, fallback_tiers = pacing.build_watchlist(
        filtered_df, today=asof_ts, history_df=df
    )

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
            "Download this data as a spreadsheet",
            data=csv_bytes,
            file_name="krannert_filtered.csv",
            mime="text/csv",
            key="filtered_csv",
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with col_note:
        st.markdown(
            '<div class="muted">The download matches the date range and reporting date you have set.</div>',
            unsafe_allow_html=True,
        )

    render_price_code_guide(df)

    # --- 1) Event Pacing Watchlist (move back up) ---
    st.subheader("Which upcoming shows need attention?", anchor="watchlist")
    st.caption(
        f"Ticket sales are counted through {asof_ts:%B %d, %Y}. Each show is compared with "
        "similar shows from past seasons at the same point before their date."
    )
    render_watchlist(watch_table, fallback_tiers, filters_summary, watch_summary)

    # --- 2) Booking Window Pacing ---
    st.subheader("When do people buy their tickets?", anchor="booking")
    with st.expander("How to read"):
        st.markdown(
            "This shows how ticket sales usually build in the four months before a show.\n\n"
            "- Read right to left: the show date is at the far right.\n"
            "- The **black line** is the typical pace across past shows.\n"
            "- The **shaded band** is the range most shows fall inside.\n"
            "- The **dashed line** is your upcoming shows. When it sits below the black line, "
            "those shows are selling slower than usual for that point in time."
        )

    mode = "Global"
    baseline_curve, cohort_label, cohort_n = build_baseline_curve(df, asof_ts, filters, mode)
    current_curve = build_current_curve(filtered_df, asof_ts)

    if baseline_curve is None or baseline_curve.empty:
        st.info("Not enough past sales to show a typical booking pattern.")
    else:
        plotly_config = {"displaylogo": False, "modeBarButtonsToRemove": ["toImage"]}
        pacing_fig = style_mono(booking_window_fig(baseline_curve, current_curve))
        st.plotly_chart(pacing_fig, use_container_width=True, config=plotly_config)
        st.caption(
            f"The typical pace is based on {cohort_n:,} past shows. "
            "The dashed line appears when at least three of your upcoming shows have sales."
        )

    # --- 3) Sales Timing Heatmap (directly under booking) ---
    st.subheader("How far ahead do people book?", anchor="timing")
    st.caption("Darker squares are where most ticket sales happen. Rows are the day of the week the ticket was bought; columns are how far ahead of the show it was bought.")
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
        st.info("Not enough sales in this view to show a booking pattern. Try widening the date range.")
    else:
        style_mono(fig_heat)
        st.plotly_chart(fig_heat, use_container_width=True, config=no_zoom_config)

    # --- 4) Event Categories: Pre vs Post COVID ---
    st.subheader("Which kinds of shows recovered after COVID?", anchor="categories")
    st.caption("Total tickets sold by category. \"Before\" covers sales up to March 2020; \"after\" covers sales from July 2021 onward.")
    with st.expander("How to read"):
        st.markdown(
            "Compare each category's ticket sales before and after the shutdown. "
            "A much shorter bar on the right means that category has not brought its audience back yet."
        )
    plotly_config = no_zoom_config
    cat_pre = fig_categories_pre(df)
    cat_post = fig_categories_post(df)
    cat_compare = fig_top_categories_pre_post(df, top_n=6)
    cols_cat = st.columns(2)
    with cols_cat[0]:
        if cat_pre is None:
            st.info("No sales from before COVID in this view.")
        else:
            style_mono(cat_pre)
            st.plotly_chart(cat_pre, use_container_width=True, config=plotly_config)
    with cols_cat[1]:
        if cat_post is None:
            st.info("No sales from after COVID in this view.")
        else:
            style_mono(cat_post)
            st.plotly_chart(cat_post, use_container_width=True, config=plotly_config)

    st.markdown("**The six biggest categories, side by side**")
    if cat_compare is None:
        st.info("Not enough data to compare categories.")
    else:
        style_mono(cat_compare)
        st.plotly_chart(cat_compare, use_container_width=True, config=plotly_config)

    # --- 5) Top Events: Pre vs Post COVID ---
    st.subheader("How did returning shows do?", anchor="returning")
    st.caption("Shows that ran both before and after the shutdown, so the two periods are directly comparable.")
    ev_compare = fig_top_events_pre_post(df, k=12)
    if ev_compare is None:
        st.info("No shows ran in both periods, so there is nothing to compare.")
    else:
        style_mono(ev_compare)
        st.plotly_chart(ev_compare, use_container_width=True, config=plotly_config)


if __name__ == "__main__":
    main()