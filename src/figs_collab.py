"""
Collab-matching visuals for event categories and events pre vs post COVID.
Uses Plotly to keep styling consistent with the Streamlit app.
"""
from __future__ import annotations

import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go

PRE_END = pd.Timestamp("2020-03-01")
POST_START = pd.Timestamp("2021-07-01")
DROP_OTHER = True


def _ensure(df: pd.DataFrame) -> pd.DataFrame:
    z = df.copy()
    if "sale_date" in z.columns:
        z["sale_date"] = pd.to_datetime(z["sale_date"], errors="coerce")
    if "event_date" in z.columns:
        z["event_date"] = pd.to_datetime(z["event_date"], errors="coerce")
    z = z.dropna(subset=["sale_date", "event_name", "qty_sold"])
    z["qty_sold"] = pd.to_numeric(z["qty_sold"], errors="coerce").fillna(0)
    return z


def _categorize(name: str) -> str:
    n = str(name or "").lower()
    if re.search(r"(theat(er|re)|play|drama|musical|opera|operetta|ballet|nutcracker)", n):
        return "Theatre"
    if re.search(r"(symph(ony|onic)|philharm|orchestra|concerto|chamber|baroque|quartet|classical)", n):
        return "Music—Classical"
    if re.search(r"jazz", n):
        return "Music—Jazz"
    if re.search(r"(rock|pop|folk|indie|singer[- ]songwriter)", n):
        return "Music—Folk/Rock/Pop"
    if re.search(r"(family|kids|children|youth|junior)", n):
        return "Family Fun"
    if re.search(r"\bdance\b", n):
        return "Dance"
    if re.search(r"circus", n):
        return "Circus Arts"
    if re.search(r"(lecture|talk|exhibit|symposium)", n):
        return "Lecture/Exhibit"
    if re.search(r"(festival|series|gala|residency|workshop|special)", n):
        return "Special Programs"
    if re.search(r"world|global|international|afro|latin|celtic|klezmer|tabla|sitar", n):
        return "World/Global"
    if re.search(r"\bmusic\b", n):
        return "Music"
    return "Other (Unmapped)"


def _category_totals(df: pd.DataFrame, drop_other: bool = True) -> pd.Series:
    d = _ensure(df)
    if d.empty:
        return pd.Series(dtype=float)
    d["event_category"] = d["event_name"].apply(_categorize)
    s = d.groupby("event_category", dropna=False)["qty_sold"].sum().sort_values(ascending=False)
    if drop_other and "Other (Unmapped)" in s.index:
        s = s.drop("Other (Unmapped)")
    return s


def _bar(series: pd.Series) -> go.Figure:
    """Simple bar chart without title (title added in app.py)."""
    fig = go.Figure(
        go.Bar(
            x=series.index,
            y=series.values,
            text=series.values,
            textposition="outside",
        )
    )
    fig.update_layout(
        yaxis_title="Tickets sold",
        xaxis_tickangle=-35,
        margin=dict(t=30, l=40, r=20, b=80),
    )
    fig.update_yaxes(tickformat=",")
    return fig


def fig_categories_pre(df: pd.DataFrame) -> go.Figure:
    pre = _ensure(df[df["sale_date"] < PRE_END])
    s = _category_totals(pre, drop_other=DROP_OTHER)
    return _bar(s)


def fig_categories_post(df: pd.DataFrame) -> go.Figure:
    post = _ensure(df[df["sale_date"] >= POST_START])
    s = _category_totals(post, drop_other=DROP_OTHER)
    return _bar(s)


def fig_top_categories_pre_post(df: pd.DataFrame, top_n: int = 6) -> go.Figure:
    """Grouped bar: top categories pre vs post (title added in app.py)."""
    df = _ensure(df)
    pre = _category_totals(df[df["sale_date"] < PRE_END], drop_other=DROP_OTHER)
    post = _category_totals(df[df["sale_date"] >= POST_START], drop_other=DROP_OTHER)
    cats = list(pre.head(top_n).index)
    pre_y = pre.reindex(cats).fillna(0)
    post_y = post.reindex(cats).fillna(0)

    fig = go.Figure()
    fig.add_bar(name="Pre-2020", x=cats, y=pre_y.values)
    fig.add_bar(name="Post-2021", x=cats, y=post_y.values)
    fig.update_layout(
        barmode="group",
        yaxis_title="Tickets sold",
        xaxis_tickangle=-35,
        margin=dict(t=30, l=40, r=20, b=80),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig.update_yaxes(tickformat=",")
    return fig


def fig_top_events_pre_post(df: pd.DataFrame, k: int = 12) -> go.Figure:
    """Grouped bar: top events present in both periods (title added in app.py)."""
    df = _ensure(df)
    pre = df[df["sale_date"] < PRE_END].groupby("event_name")["qty_sold"].sum()
    post = df[df["sale_date"] >= POST_START].groupby("event_name")["qty_sold"].sum()
    common = pre.index.intersection(post.index)
    pre, post = pre.loc[common], post.loc[common]
    top = (pre + post).sort_values(ascending=False).head(k).index

    fig = go.Figure()
    fig.add_bar(name="Pre-2020", x=top, y=pre.reindex(top).fillna(0).values)
    fig.add_bar(name="Post-2021", x=top, y=post.reindex(top).fillna(0).values)
    fig.update_layout(
        barmode="group",
        yaxis_title="Tickets sold",
        xaxis_tickangle=-35,
        margin=dict(t=30, l=40, r=20, b=120),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig.update_yaxes(tickformat=",")
    return fig

