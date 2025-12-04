from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import data_prep

DATA_DIR = Path(__file__).parent / "data"
ASSETS_DIR = Path(__file__).parent / "assets"
DEFAULT_DATA_PATH = DATA_DIR / "sales_2016_2026_combined.csv"
MAX_DAYS_LOOKAHEAD = 120
AHEAD_THRESHOLD = 5
BEHIND_THRESHOLD = -5
MAX_FILTER_VALUES = 3
LEAD_BUCKET_LABELS = [bucket[2] for bucket in data_prep.LEAD_BUCKETS]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
COHORT_LEVELS: list[tuple[str, tuple[str, ...]]] = [
    ("type_day_venue", ("event_type", "weekday", "venue")),
    ("type_day", ("event_type", "weekday")),
    ("type_only", ("event_type",)),
    ("global", ()),
]
COHORT_LABELS = {
    "type_day_venue": "event type + weekday + venue",
    "type_day": "event type + weekday",
    "type_only": "event type",
    "global": "all events",
    "none": "insufficient history",
}


def _format_int(value: float | int | None) -> str:
    if value is None or np.isnan(value):
        return "–"
    return f"{value:,.0f}"


def _format_currency(value: float | int | None) -> str:
    if value is None or np.isnan(value):
        return "–"
    return f"${value:,.0f}"


def _format_percent(value: float | None, decimals: int = 1) -> str:
    if value is None or np.isnan(value):
        return "–"
    return f"{value:.{decimals}f}%"


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
    categories_text = summarize(filters.get("event_categories") or [], "Category")
    channels_text = summarize(filters.get("channels") or [], "Channel")

    start, end = filters.get("date_range", (None, None))
    if start is not None and end is not None:
        date_text = f"Dates: {start.date():%b %d %Y} → {end.date():%b %d %Y}"
    else:
        date_text = "Dates: All"

    return " | ".join([seasons_text, categories_text, channels_text, date_text])


def default_filter_bounds(df: pd.DataFrame) -> dict:
    today = pd.Timestamp.today().normalize()
    defaults = {}
    if "season" in df.columns:
        seasons = sorted(df["season"].dropna().unique())
        defaults["all_seasons"] = seasons
        defaults["selected_seasons"] = seasons[-5:] if len(seasons) > 5 else seasons
    else:
        defaults["all_seasons"] = []
        defaults["selected_seasons"] = []

    min_event = pd.Timestamp("2016-01-01")
    max_event = min(df["event_date"].max(), today + pd.Timedelta(days=365)) if df["event_date"].notna().any() else today
    defaults["date_bounds"] = (min_event.date(), max_event.date())
    return defaults


def sidebar_filters(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, str]:
    defaults = default_filter_bounds(df)
    st.sidebar.subheader("Filters")

    seasons = defaults["all_seasons"]
    selected_seasons = st.sidebar.multiselect(
        "Season", seasons, default=defaults["selected_seasons"], placeholder="All seasons"
    )

    categories = sorted(df["event_category"].dropna().unique()) if "event_category" in df.columns else []
    selected_categories = st.sidebar.multiselect(
        "Event category", categories, default=categories, placeholder="All categories"
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
    if selected_categories:
        mask &= df["event_category"].isin(selected_categories)
    if selected_channels and "channel" in df.columns:
        mask &= df["channel"].isin(selected_channels)
    mask &= df["event_date"].between(start_ts, end_ts)

    filtered = df.loc[mask].copy()
    filters = {
        "seasons": selected_seasons,
        "event_categories": selected_categories,
        "channels": selected_channels,
        "date_range": (start_ts, end_ts),
    }
    return filtered, filters, summarize_filters(filters)


def pace_status(gap: float) -> str:
    if np.isnan(gap):
        return "No benchmark"
    if gap >= AHEAD_THRESHOLD:
        return "Ahead"
    if gap <= BEHIND_THRESHOLD:
        return "Behind"
    return "On pace"


def lookup_median_value(
    cohort_tables: dict[str, pd.DataFrame], event_type: str, weekday: str, venue: str, lead_days: int
) -> tuple[float, str]:
    metadata = {"event_type": event_type, "weekday": weekday, "venue": venue}
    lead_days = int(np.clip(lead_days, 0, MAX_DAYS_LOOKAHEAD))
    for level_name, cols in COHORT_LEVELS:
        table = cohort_tables.get(level_name)
        if table is None or table.empty:
            continue
        subset = table
        for col in cols:
            subset = subset[subset[col] == metadata[col]]
        subset = subset[subset["lead_days"] == lead_days]
        subset = subset[subset["sample_size"] >= 20]
        if not subset.empty:
            return float(subset["median_cum_pct"].iloc[0]), level_name
    return np.nan, "none"


def build_watchlist(base: pd.DataFrame, cohort_tables: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict, set[str]]:
    today = pd.Timestamp.today().normalize()
    window_mask = (base["event_date"] >= today) & (base["event_date"] <= today + pd.Timedelta(days=MAX_DAYS_LOOKAHEAD))
    scope = base[window_mask]
    summary = {"behind": 0, "evaluated": 0}
    if scope.empty:
        cols = ["Event", "Days-out", "Sold so far", "Cum %", "Median @D", "Gap (pp)", "Status", "Event total"]
        return pd.DataFrame(columns=cols), summary, set()

    totals = base.groupby("event_instance")["qty_sold"].sum()
    sold_to_date = base[base["sale_date"] <= today].groupby("event_instance")["qty_sold"].sum()
    meta = (
        scope.groupby("event_instance")
        .agg(
            event_date=("event_date", "max"),
            event_name=("event_label", "first"),
            event_type=("event_type", "first"),
            weekday=("weekday", "first"),
            venue=("venue", "first"),
        )
        .reset_index()
    )
    meta["days_out"] = (meta["event_date"] - today).dt.days
    meta = meta[meta["days_out"].between(0, MAX_DAYS_LOOKAHEAD)]
    meta["event_total"] = meta["event_instance"].map(totals).fillna(0)
    meta["sold_so_far"] = meta["event_instance"].map(sold_to_date).fillna(0)
    meta = meta[meta["event_total"] > 0]
    meta["cum_pct"] = np.clip((meta["sold_so_far"] / meta["event_total"]) * 100, 0, 100)

    records = []
    fallback_levels: set[str] = set()
    for row in meta.itertuples():
        median_pct, level_used = lookup_median_value(
            cohort_tables, row.event_type, row.weekday, row.venue, int(row.days_out)
        )
        fallback_levels.add(level_used)
        gap = row.cum_pct - median_pct
        status = pace_status(gap)
        summary["evaluated"] += 1
        if status == "Behind":
            summary["behind"] += 1
        records.append(
            {
                "Event": row.event_name,
                "Days-out": int(row.days_out),
                "Sold so far": int(row.sold_so_far),
                "Cum %": row.cum_pct,
                "Median @D": median_pct,
                "Gap (pp)": gap,
                "Status": status,
                "Event total": int(row.event_total),
            }
        )

    table = pd.DataFrame(records).sort_values("Gap (pp)").head(50)
    action_text = "Recommendation: push social/email; adjust price band"
    table["Action"] = np.where(table["Status"] == "Behind", action_text, "")
    return table, summary, fallback_levels


def compute_kpis(base: pd.DataFrame, watch_summary: dict, today: pd.Timestamp) -> dict:
    last_30_mask = (base["sale_date"] >= today - pd.Timedelta(days=29)) & (base["sale_date"] <= today)
    prev_30_mask = (base["sale_date"] >= today - pd.Timedelta(days=59)) & (base["sale_date"] < today - pd.Timedelta(days=29))

    tickets_30d = base.loc[last_30_mask, "qty_sold"].sum()
    tickets_prev = base.loc[prev_30_mask, "qty_sold"].sum()
    delta_30d = (tickets_30d - tickets_prev) / tickets_prev if tickets_prev else None

    amount_col = base["amount"] if "amount" in base.columns else pd.Series(dtype=float)
    amount_30d = amount_col.loc[last_30_mask].sum()
    qty_30d = base.loc[last_30_mask, "qty_sold"].sum()
    avg_price_30d = amount_30d / qty_30d if qty_30d else None

    open_events = base[(base["event_date"] >= today) & (base["event_date"] <= today + pd.Timedelta(days=30))][
        "event_instance"
    ].nunique()

    evaluated = watch_summary.get("evaluated", 0)
    behind = watch_summary.get("behind", 0)
    behind_pct = (behind / evaluated) * 100 if evaluated else 0

    return {
        "tickets_30d": tickets_30d,
        "delta_30d": delta_30d,
        "avg_price_30d": avg_price_30d,
        "open_events": open_events,
        "behind_pct": behind_pct,
        "behind_detail": f"{behind} of {evaluated}" if evaluated else "–",
    }


def render_kpis(kpis: dict) -> None:
    col1, col2, col3, col4 = st.columns(4)
    delta_display = _format_percent(kpis["delta_30d"] * 100 if kpis["delta_30d"] is not None else None)
    col1.metric("Tickets last 30 days", _format_int(kpis["tickets_30d"]), delta=delta_display)
    col2.metric("Avg price last 30 days", _format_currency(kpis["avg_price_30d"]))
    col3.metric("Open events next 30 days", _format_int(kpis["open_events"]))
    col4.metric("% events behind pace (≤120d)", _format_percent(kpis["behind_pct"]), kpis["behind_detail"])


def event_watchlist_fig(table: pd.DataFrame) -> go.Figure | None:
    if table.empty:
        return None
    display = table.copy()
    display["Cum %"] = display["Cum %"].round(1)
    display["Median @D"] = display["Median @D"].round(1)
    display["Gap (pp)"] = display["Gap (pp)"].round(1)
    values = [display[col].tolist() for col in display.columns if col not in {"Event total"}]
    headers = [col for col in display.columns if col not in {"Event total"}]
    formats = [None, ",d", ",d", ".1f", ".1f", ".1f", None, None]
    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=headers,
                    fill_color="#1f77b4",
                    font=dict(color="white", size=13),
                    align="left",
                ),
                cells=dict(
                    values=values,
                    align="left",
                    format=formats,
                    height=28,
                    fill_color="#f7f9fc",
                    font=dict(color="#0f1116", size=12),
                    line_color="#d8e3ef",
                ),
            )
        ]
    )
    fig.update_layout(template="plotly_white", margin=dict(l=0, r=0, t=10, b=0), height=460)
    return fig


def booking_window_fig(pacing_df: pd.DataFrame, current_curve: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if pacing_df.empty:
        fig.update_layout(title="Pacing curve unavailable", template="plotly_white")
        return fig

    fig.add_trace(
        go.Scatter(
            x=pacing_df["lead_days"],
            y=pacing_df["p75_cum_pct"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=pacing_df["lead_days"],
            y=pacing_df["p25_cum_pct"],
            fill="tonexty",
            mode="lines",
            line=dict(width=0),
            fillcolor="rgba(31,119,180,0.15)",
            name="Historic IQR (p25–p75)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=pacing_df["lead_days"],
            y=pacing_df["median_cum_pct"],
            mode="lines",
            line=dict(color="#1f77b4", width=3),
            name="Historic median",
        )
    )
    if not current_curve.empty:
        fig.add_trace(
            go.Scatter(
                x=current_curve["lead_days"],
                y=current_curve["cum_pct"],
                mode="lines",
                line=dict(color="#ff7f0e", width=3, dash="dash"),
                name="Current selection",
            )
        )

    fig.update_layout(
        title="Booking window pacing vs historical median",
        xaxis_title="Days before event (120 → 0)",
        yaxis_title="Cumulative % of tickets sold",
        template="plotly_white",
        legend_orientation="h",
        legend_yanchor="bottom",
        legend_y=1.02,
        legend_x=0,
    )
    fig.update_yaxes(range=[0, 105], ticksuffix="%", tickformat=".0f")
    fig.update_xaxes(autorange="reversed", range=[120, 0])
    return fig


def avg_tickets_fig(avg_df: pd.DataFrame, share_df: pd.DataFrame) -> go.Figure | None:
    if avg_df.empty:
        return None
    share_threshold = 0.05
    share_lookup = share_df.set_index("event_type")["share_ratio"] if not share_df.empty else pd.Series(dtype=float)
    kept = avg_df[avg_df["event_type"].apply(lambda x: share_lookup.get(x, 0) >= share_threshold or x != "Other (Unmapped)")]
    if kept.empty:
        kept = avg_df
    fig = px.bar(
        kept,
        x="event_type",
        y="avg_tickets",
        color="window",
        barmode="group",
        text_auto=".0f",
        title="Avg tickets per event (pre vs post)",
    )
    fig.update_layout(template="plotly_white", legend_title="")
    fig.update_yaxes(title="Avg tickets per event", tickformat=",")
    fig.update_xaxes(title="Event type")
    return fig


def share_shift_fig(pre: pd.DataFrame, post: pd.DataFrame) -> go.Figure | None:
    if pre.empty and post.empty:
        return None
    pre = pre.assign(window="Pre")
    post = post.assign(window="Post")
    combined = pd.concat([pre, post], ignore_index=True)
    if combined.empty:
        return None
    combined["share_pct"] = combined.groupby("window")["share_ratio"].transform(lambda s: (s / s.sum()) * 100 if s.sum() else 0)
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
    return fig


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
        title="Sales calendar heatmap (weekday × lead bucket)",
    )
    fig.update_layout(template="plotly_white", coloraxis_colorbar_title="% of weekday tickets", coloraxis_colorbar_tickformat=".0%")
    fig.update_traces(hovertemplate="Weekday: %{y}<br>Lead bucket: %{x}<br>Share: %{z:.1%}<extra></extra>")
    fig.update_yaxes(title="Weekday")
    fig.update_xaxes(title="Lead bucket")
    return fig


def checkpoint_summary(table: pd.DataFrame, target_day: int, tolerance: int = 2) -> dict | None:
    if table.empty:
        return None
    matches = table[table["Days-out"].between(target_day - tolerance, target_day + tolerance)]
    if matches.empty:
        return None
    weights = matches["Event total"].clip(lower=1)
    actual = np.average(matches["Cum %"], weights=weights)
    median = np.average(matches["Median @D"], weights=weights)
    gap = actual - median
    return {
        "label": f"D-{target_day}",
        "actual": actual,
        "median": median,
        "gap": gap,
        "status": pace_status(gap),
    }


def render_checkpoint_cards(table: pd.DataFrame) -> None:
    summaries = [checkpoint_summary(table, 30), checkpoint_summary(table, 7)]
    valid = [s for s in summaries if s is not None]
    if not valid:
        st.info("Not enough upcoming events to compute early/late checkpoints.")
        return
    cols = st.columns(len(valid))
    for col, summary in zip(cols, valid):
        col.metric(
            f"{summary['label']} cumulative %",
            _format_percent(summary["actual"]),
            f"Median {summary['median']:.1f}%",
        )
        col.caption(f"Status: {summary['status']} ({summary['gap']:.1f} percentage points)")


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
        st.download_button("Download PNG", data=png_bytes, file_name=f"{key}.png", mime="image/png", key=f"{key}_png")
    except Exception as exc:
        st.warning(f"PNG export unavailable: {exc}")


def main() -> None:
    st.set_page_config(page_title="Krannert Dashboard", layout="wide")
    st.title("Krannert Dashboard")

    st.sidebar.header("Data source")
    uploaded = st.sidebar.file_uploader("Upload a CSV", type=["csv"])
    local_path = st.sidebar.text_input(
        "Or load from local path",
        value=str(DEFAULT_DATA_PATH),
        help="Relative or absolute path to a CSV (default points to ./data).",
    )

    df, source_label = load_dataset(uploaded, local_path)
    if df is None:
        has_local = any(DATA_DIR.glob("*.csv"))
        if uploaded is None and not has_local:
            st.info("No CSV detected; showing synthetic sample data for preview.")
            df = data_prep.make_fake_data()
            source_label = "Synthetic sample data"
        else:
            st.info("Provide a CSV via upload or local path to explore performance.")
            return

    st.caption(f"Data source: {source_label} • Rows: {len(df):,}")

    filtered_df, filters, filters_summary = sidebar_filters(df)
    if filtered_df.empty:
        st.warning("No rows match the current filters.")
        return

    aggregates = data_prep.derive_core(filtered_df)
    base = aggregates["base"]
    cohort_tables = aggregates["cohort_library"]
    today = pd.Timestamp.today().normalize()

    watch_table, watch_summary, fallback_levels = build_watchlist(base, cohort_tables)
    kpis = compute_kpis(base, watch_summary, today)
    render_kpis(kpis)

    csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download filtered CSV",
        data=csv_bytes,
        file_name="krannert_filtered.csv",
        mime="text/csv",
        key="filtered_csv",
    )

    watch_fig = event_watchlist_fig(watch_table)
    render_plot("Event pacing watchlist (next 120 days)", watch_fig, key="event_status", subtitle=filters_summary)
    if "type_day" in fallback_levels or "type_only" in fallback_levels or "global" in fallback_levels:
        fallback_msg = ", ".join(COHORT_LABELS[level] for level in fallback_levels if level != "type_day_venue")
        st.caption(f"Cohort fallback due to limited history: {fallback_msg}.")

    current_curve = pd.DataFrame()
    event_curves = aggregates.get("event_curves", pd.DataFrame())
    if not event_curves.empty:
        current_curve = (
            event_curves[event_curves["event_date"] >= today]
            .groupby("lead_days", as_index=False)["cum_pct"]
            .mean()
        )
    pacing_fig = booking_window_fig(aggregates.get("pacing_curves", pd.DataFrame()), current_curve)
    render_plot("1) Booking window pacing", pacing_fig, key="pacing_curve", subtitle="Historic median vs current selection")

    avg_tickets_fig_obj = avg_tickets_fig(aggregates["avg_tickets"], aggregates["category_share_post"])
    render_plot("2) Avg tickets per event (pre vs post)", avg_tickets_fig_obj, key="avg_tickets", subtitle=filters_summary)

    share_fig = share_shift_fig(aggregates["category_share_pre"], aggregates["category_share_post"])
    render_plot("3) Category share shift", share_fig, key="category_share", subtitle=filters_summary)

    render_plot("4) Top events: pre vs post", top_events_fig(aggregates["events_pre_post"]), key="top_events", subtitle=filters_summary)

    render_plot("5) Sales calendar heatmap", sales_heatmap_fig(aggregates["dows_heatmap"]), key="calendar_heatmap", subtitle=filters_summary)

    render_checkpoint_cards(watch_table)


if __name__ == "__main__":
    main()

