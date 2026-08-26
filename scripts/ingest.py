#!/usr/bin/env python3
"""
Combine Krannert per-season daily-sales exports into the dashboard's dataset.

Usage
-----
    # add one new season to the existing combined file
    python scripts/ingest.py data/raw/"Daily Sales 26-27 Season.csv"

    # rebuild the combined file from every raw export
    python scripts/ingest.py data/raw/*.csv --rebuild

Each raw file is expected to be one season of daily sales with these columns
(header matching is case/space-insensitive, extra columns are ignored):

    sale_date, qty_sold, amount, event_part, event_name, event_date

`season` and `source_file` are derived automatically — season comes from the
filename (e.g. "Daily Sales 26-27 Season.csv" -> "2026-2027"), falling back to
the sale dates in the file.

The script validates before writing and refuses to write if anything looks
structurally wrong. Data-quality warnings (implausible dates, refunds, blank
names) are reported but do not block, since the dashboard already skips those
rows; review them and fix at the source when possible.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
COMBINED = REPO / "data" / "sales_2016_2026_combined.csv"

CANONICAL_COLUMNS = [
    "sale_date",
    "qty_sold",
    "amount",
    "event_part",
    "event_name",
    "event_date",
    "season",
    "source_file",
]
REQUIRED_INPUT = ["sale_date", "qty_sold", "event_name", "event_date"]

# Header spellings seen in raw exports -> canonical name
ALIASES = {
    "sale date": "sale_date",
    "saledate": "sale_date",
    "date": "sale_date",
    "transaction_date": "sale_date",
    "qty": "qty_sold",
    "quantity": "qty_sold",
    "tickets": "qty_sold",
    "tickets_sold": "qty_sold",
    "seats": "qty_sold",
    "total": "amount",
    "revenue": "amount",
    "sales": "amount",
    "price": "amount",
    "performance": "event_name",
    "event": "event_name",
    "production": "event_name",
    "perf_date": "event_date",
    "performance_date": "event_date",
    "show_date": "event_date",
    "part": "event_part",
    "price_type": "event_part",
    "ticket_type": "event_part",
    "price_category": "event_part",
}

PLAUSIBLE_MIN = pd.Timestamp("2000-01-01")
PLAUSIBLE_MAX = pd.Timestamp.today() + pd.Timedelta(days=1460)  # ~4 years ahead


def normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = []
    for col in df.columns:
        key = str(col).strip().lower().replace(" ", "_")
        cleaned.append(ALIASES.get(key, ALIASES.get(key.replace("_", " "), key)))
    df.columns = cleaned
    return df


def season_from_filename(path: Path) -> str | None:
    """'Daily Sales 26-27 Season.csv' -> '2026-2027'."""
    m = re.search(r"(\d{2})\s*-\s*(\d{2})", path.stem)
    if not m:
        return None
    start, end = int(m.group(1)), int(m.group(2))
    return f"20{start:02d}-20{end:02d}"


def season_from_dates(sale_dates: pd.Series) -> str:
    """Krannert seasons run Jul->Jun; label by the July-start year."""
    median = sale_dates.dropna().median()
    if pd.isna(median):
        return "unknown"
    start = median.year if median.month >= 7 else median.year - 1
    return f"{start}-{start + 1}"


def load_raw(path: Path) -> pd.DataFrame:
    df = normalize_headers(pd.read_csv(path))

    missing = [c for c in REQUIRED_INPUT if c not in df.columns]
    if missing:
        raise SystemExit(
            f"ERROR {path.name}: missing required column(s): {', '.join(missing)}\n"
            f"       found: {', '.join(df.columns)}\n"
            f"       expected (any casing): {', '.join(REQUIRED_INPUT)}"
        )

    for col in ("qty_sold", "amount"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("sale_date", "event_date"):
        df[col] = pd.to_datetime(df[col], errors="coerce")

    if "amount" not in df.columns:
        df["amount"] = 0.0
    if "event_part" not in df.columns:
        df["event_part"] = ""

    df["event_name"] = (
        df["event_name"].astype(str).str.strip().replace({"nan": "", "NaN": "", "None": ""})
    )
    df["event_part"] = df["event_part"].astype(str).str.strip().str.upper().replace({"NAN": ""})
    df["season"] = season_from_filename(path) or season_from_dates(df["sale_date"])
    df["source_file"] = path.name

    return df[CANONICAL_COLUMNS]


def report_quality(df: pd.DataFrame, label: str) -> None:
    """Print data-quality warnings. These never block the write."""
    notes = []

    unparsed = df["sale_date"].isna().sum() + df["event_date"].isna().sum()
    if unparsed:
        notes.append(f"{unparsed} row(s) with unreadable dates (dropped by the dashboard)")

    blank = (df["event_name"].isna() | (df["event_name"].str.len() == 0)).sum()
    if blank:
        notes.append(f"{blank} row(s) with a blank event name (dropped by the dashboard)")

    implausible = df[
        df["event_date"].notna()
        & ((df["event_date"] < PLAUSIBLE_MIN) | (df["event_date"] > PLAUSIBLE_MAX))
    ]
    if len(implausible):
        names = implausible.groupby(["event_name", "event_date"]).size()
        notes.append(f"{len(implausible)} row(s) with an implausible event_date — likely typos:")
        for (name, date), n in names.items():
            notes.append(f"      · {name} -> {date:%Y-%m-%d} ({n} rows)")

    late = df[
        df["sale_date"].notna()
        & df["event_date"].notna()
        & (df["sale_date"] > df["event_date"] + pd.Timedelta(days=1))
    ]
    if len(late):
        top = late.groupby("event_name").size().sort_values(ascending=False).head(3)
        notes.append(
            f"{len(late)} row(s) sold after the event date "
            "(normal for subscriptions; otherwise check the event_date):"
        )
        for name, n in top.items():
            notes.append(f"      · {name} ({n} rows)")

    refunds = (df["qty_sold"] <= 0).sum()
    if refunds:
        notes.append(f"{refunds} row(s) with qty_sold <= 0 (refunds/exchanges — kept, they net out)")

    dupes = df.duplicated().sum()
    if dupes:
        notes.append(f"{dupes} exact duplicate row(s) — removed")

    if notes:
        print(f"\n  Data-quality notes for {label}:")
        for n in notes:
            print(f"    ! {n}" if not n.startswith("      ") else n)
    else:
        print(f"\n  No data-quality issues found in {label}.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", type=Path, help="raw per-season CSV export(s)")
    ap.add_argument("--rebuild", action="store_true",
                    help="rebuild the combined file from the given files only (default: append)")
    ap.add_argument("--out", type=Path, default=COMBINED, help=f"output path (default: {COMBINED})")
    ap.add_argument("--dry-run", action="store_true", help="validate and report; write nothing")
    args = ap.parse_args()

    incoming = []
    for path in args.files:
        if not path.exists():
            raise SystemExit(f"ERROR: no such file: {path}")
        df = load_raw(path)
        print(f"\nRead {path.name}: {len(df):,} rows, season {df['season'].iloc[0]}")
        report_quality(df, path.name)
        incoming.append(df)

    new = pd.concat(incoming, ignore_index=True)

    if args.rebuild or not args.out.exists():
        combined = new
        print(f"\nRebuilding {args.out.name} from {len(args.files)} file(s).")
    else:
        existing = pd.read_csv(args.out)
        existing["sale_date"] = pd.to_datetime(existing["sale_date"], errors="coerce")
        existing["event_date"] = pd.to_datetime(existing["event_date"], errors="coerce")
        replacing = set(new["source_file"]) & set(existing["source_file"])
        if replacing:
            print(f"\nReplacing existing rows from: {', '.join(sorted(replacing))}")
            existing = existing[~existing["source_file"].isin(replacing)]
        combined = pd.concat([existing[CANONICAL_COLUMNS], new], ignore_index=True)

    before = len(combined)
    combined = combined.drop_duplicates()
    if before != len(combined):
        print(f"  Removed {before - len(combined):,} duplicate row(s).")

    combined = combined.sort_values(["sale_date", "event_name"], kind="stable")

    # Structural guard: never write a file the dashboard can't read
    usable = combined.dropna(subset=["sale_date", "event_date", "event_name", "qty_sold"])
    if usable.empty:
        raise SystemExit("ERROR: no usable rows after cleaning — refusing to write.")
    if usable["qty_sold"].sum() <= 0:
        raise SystemExit("ERROR: total tickets sold is zero — check the qty column mapping.")

    print(f"\nCombined total: {len(combined):,} rows across {combined['season'].nunique()} seasons")
    print(f"  Seasons: {', '.join(sorted(combined['season'].dropna().unique()))}")
    print(f"  Sales:   {combined['sale_date'].min():%Y-%m-%d} -> {combined['sale_date'].max():%Y-%m-%d}")
    print(f"  Usable:  {len(usable):,} rows ({len(combined) - len(usable):,} skipped by the dashboard)")

    if args.dry_run:
        print("\nDry run — nothing written.")
        return 0

    combined["sale_date"] = combined["sale_date"].dt.strftime("%Y-%m-%d")
    combined["event_date"] = combined["event_date"].dt.strftime("%Y-%m-%d")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.out, index=False)
    print(f"\nWrote {args.out}")
    print("Next: commit and push, and the live dashboard redeploys automatically.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
