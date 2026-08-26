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

Use `data/template_sales_import.csv` as the spreadsheet template — one row per sale
transaction, same columns as the combined file:

`sale_date, qty_sold, amount, event_part, event_name, event_date, season, source_file`

- `event_part` is the price-category code (SA, SC, UI, YT, C, FA, SU, ...).
- Dates should be `YYYY-MM-DD`.
- `season` is the label like `2026-2027`; `source_file` is just a note about where the rows came from.

Two ways to load it:

1. **Quick look (temporary):** export the new season's sales into the template columns,
   append them to a copy of the combined CSV, and upload it via the sidebar's
   "Upload updated data". This affects only your browser session.
2. **Permanent update (live site):** append the new rows to
   `data/sales_2016_2026_combined.csv` in this repo and push to `main` — Streamlit
   Cloud redeploys automatically. The dashboard's "as-of" date defaults to the newest
   `sale_date` in the file, so pacing updates on its own.

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

