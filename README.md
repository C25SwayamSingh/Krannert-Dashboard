# krannert-dash

**🔗 Live app:** [krannert-dashboard-datastorytelling.streamlit.app](https://krannert-dashboard-datastorytelling.streamlit.app/)

Streamlit dashboard for analyzing Krannert event sales with a lightweight data prep layer and Plotly visualizations.

## Project structure

- `app.py` – Streamlit entrypoint with file uploader / local file selector
- `src/data_prep.py` – load → clean → aggregate pipeline (function-only skeleton)
- `src/utils.py` – shared helpers (category mapping, date logic, caching helpers)
- `assets/` – exported figures or brand assets
- `data/` – drop source CSVs here if not using the uploader
- `tests/` – smoke tests to guard critical paths

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## How to run locally

1. Activate the virtual environment (`source .venv/bin/activate`).
2. Launch the dashboard:
   ```bash
   streamlit run app.py
   ```
3. Load data either by uploading a CSV via the sidebar or by picking a file from `./data`.
4. If no CSVs are present, the app automatically spins up synthetic sample data so you can preview the visuals.

## Adding your CSVs

- Drop files under `data/` (e.g., `data/sales_2016_2026_combined.csv`) so they appear in the sidebar picker.
- Filenames can be anything; only the required columns matter.

## Updating with new season data

The dashboard reads one combined file, `data/sales_2016_2026_combined.csv`.
`scripts/ingest.py` builds that file from Krannert's raw per-season exports, so
nobody has to hand-clean or hand-merge anything.

### What Krannert sends

One CSV per season, exported the same way as the existing
`Daily Sales 24-25 Season.csv` files — **no cleaning needed**, one row per sale
transaction. Name the file with the season in it (`Daily Sales 26-27 Season.csv`)
and the script derives the season label automatically.

Required columns (header matching is case- and spacing-insensitive, so
`Sale Date`, `sale_date`, and `SALE DATE` all work; extra columns are ignored):

| Column | Also accepted as | Notes |
|---|---|---|
| `sale_date` | Sale Date, Date, Transaction Date | `YYYY-MM-DD` preferred |
| `qty_sold` | Quantity, Qty, Tickets, Seats | negatives = refunds, kept |
| `event_name` | Performance, Event, Production | e.g. `Fr/The Nutcracker` |
| `event_date` | Performance Date, Show Date, Perf Date | the performance date |
| `amount` | Total, Revenue, Sales, Price | optional; defaults to 0 |
| `event_part` | Price Type, Ticket Type, Price Category | optional; the SA/UI/SC/YT price code |

`data/template_sales_import.csv` is a filled-in example of this layout.

### Running the pipeline

```bash
python scripts/ingest.py "data/raw/Daily Sales 26-27 Season.csv"
```

This validates the file, appends it to the combined CSV, and reports any data
quality issues. Useful flags:

- `--dry-run` — validate and report without writing (always worth doing first)
- `--rebuild` — rebuild the combined file from scratch: `python scripts/ingest.py data/raw/*.csv --rebuild`

Re-running with a season that is already present **replaces** those rows rather
than duplicating them, so it is safe to re-ingest a corrected export.

Then commit and push; Streamlit Cloud redeploys automatically and the "as-of"
date follows the newest `sale_date` in the file, so pacing updates by itself.

For a one-off look without touching the repo, the sidebar's
"Upload updated data" accepts an already-combined CSV for that browser session only.

### What the script checks

Structural problems (missing columns, an unreadable file, zero usable rows) stop
the run with an explicit message. Data-quality problems are reported but do not
block, because the dashboard already skips unusable rows:

- unreadable or blank dates and event names
- implausible `event_date` values — typos like a 1930 or 2007 performance date
- sales dated after the performance (normal for subscriptions, otherwise a bad `event_date`)
- `qty_sold <= 0` refund/exchange rows (kept — they net out correctly)
- exact duplicate rows (removed)

### Known issues in the current dataset

Found by running these checks over the existing combined file — worth correcting
at the source when Krannert next exports:

- `Sa/Cabaret` has `event_date` **1930-03-07** (181 rows) and
  `Tu/Russian National Ballet Theatre: Carmen/Romeo and Juliet` has **2007-01-17**
  (236 rows). Both are clearly year typos, and both fall outside the dashboard's
  date range, so those rows are currently invisible in every chart.
- `Sa/Oklahoma!` is dated 2024-06-16 but sold Aug–Nov 2024 — likely 2025-06-16.
- 7 rows have no `event_name`/`event_date` and are skipped on load.
- 106 rows have `qty_sold <= 0` (refunds); these are intentional and net out.

## Data expectations

Input CSV must include (case-insensitive) columns. Extra columns are ignored.

- `sale_date` (date/datetime)
- `event_date` (date/datetime)
- `event_name` (string)
- `event_part` (optional string)
- `qty_sold` (numeric)
- `amount` (optional numeric, used for revenue)
- `channel` (optional string)
- `customer_type` (optional string)
- `season` (optional string)

Pre window: `sale_date < 2020-03-01`. Post window: `sale_date >= 2021-07-01`.

## Exporting figures

- Every Plotly chart renders with a "Download PNG" button (using Kaleido).
- Clicking the button saves a PNG locally and also writes the file into `assets/`.
- You can also click "Download filtered CSV" beneath the KPIs for the current-filter dataset.

## Running tests

```bash
source .venv/bin/activate
pytest
```

## Optional deployment (Streamlit Community Cloud)

1. Push this repo to GitHub.
2. In [Streamlit Community Cloud](https://streamlit.io/cloud), select "New app".
3. Choose the repo + main branch and set the entrypoint to `app.py`.
4. The platform installs `requirements.txt` automatically, so no Procfile is needed. Optional secrets (e.g., `ST_AUTH_TOKEN`) can be managed via Streamlit's settings UI.

