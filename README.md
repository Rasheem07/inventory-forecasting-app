# Stockroom — inventory demand dashboard

FastAPI backend and a single-page Streamlit dashboard with locally compiled Tailwind CSS and interactive Plotly charts. Uses the included `train.csv` (10 stores, 50 products) rather than generated data.

## Run

Python 3.10+:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python run.py
```

- Dashboard: http://127.0.0.1:8501
- Interactive API documentation: http://127.0.0.1:8000/docs

The launcher starts both services, waits for the API to become ready, and shuts them down on Ctrl+C. Both bind to localhost. `API_PORT` and `DASHBOARD_PORT` override their ports.

Or run separately in two terminals, from the project root:

```bash
.venv/bin/python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
.venv/bin/python -m streamlit run dashboard/app.py
```

Set `INVENTORY_API_URL` for a separately hosted API; set `INVENTORY_DATA_PATH` to replace the training CSV. Restart the API after changing the dataset. No API keys or external model service are required. The API is read-only and unauthenticated; add access controls before exposing private datasets publicly.

## Deploy on Streamlit Community Cloud

In Streamlit Community Cloud, create an app with:

- Repository: `Rasheem07/inventory-forecasting-app`
- Branch: `main`
- Main file path: `streamlit_app.py`
- Python: `3.12`

The root `requirements.txt` declares dependencies and the training data and compiled CSS are included. No secrets, API keys or Node build step are needed. Leave `INVENTORY_API_URL` unset: standalone mode loads the shared forecasting service inside Streamlit and caches it in memory, so there is no localhost API dependency. The FastAPI endpoints remain available when deployed separately or started locally with `run.py`.

To run the same standalone mode locally:

```bash
.venv/bin/python -m streamlit run streamlit_app.py
```

To use a separately deployed API instead, set `INVENTORY_API_URL` to its HTTPS base URL in the Streamlit server environment. The sidebar **Refresh data** button refreshes dashboard responses; restart the app to reload a changed dataset or name catalog.

## Dashboard

- Multi-select store and product filters, preset/custom history dates, and 7–90 day forecast horizon (default 30).
- Sales totals, average demand, forecast units, and next-30-day change.
- Daily sales and forecast chart with an optional indicative error range.
- Store comparisons, top products, weekday patterns, and held-out prediction chart.
- Sortable store–product planning table, table CSV, and daily store–product forecast CSV.
- API connection failures and empty/invalid date selections produce actionable messages.

Tailwind utilities style the custom HTML, with additional CSS for native Streamlit widgets. Compiled CSS is included, so Node is unnecessary to run. To rebuild styles after changing utility classes:

```bash
npm ci
npm run build:css
```

## API

| Endpoint | Returns |
| --- | --- |
| `GET /api/health` | Readiness |
| `GET /api/metadata` | Available IDs, date coverage, model details |
| `GET /api/dashboard` | All metrics, charts and per-series forecasts |
| `GET /api/history` | Filtered daily sales |
| `GET /api/forecast` | Aggregate daily predictions, ranges, validation |
| `GET /api/breakdown/store` | Store totals |
| `GET /api/breakdown/product` | Product totals |
| `GET /api/forecast.csv` | Daily predictions per store/product |

Analytics and export routes accept the same filters: repeated `stores` / `products` IDs, `start_date`, `end_date` (ISO dates), and `horizon` (1–90). Omitted/empty ID filters include all IDs. Invalid IDs, reversed dates and dates outside history return 422.

```bash
curl 'http://127.0.0.1:8000/api/dashboard?stores=1&stores=2&products=3&horizon=30&start_date=2017-10-01&end_date=2017-12-31'
curl -o forecast.csv 'http://127.0.0.1:8000/api/forecast.csv?stores=1&products=3&horizon=30'
```

History dates affect historical charts/totals only. Forecasts always begin immediately after the last dataset date, not after the selected history end or the current calendar date. The included data ends **2017-12-31**, so forecasts begin **2018-01-01**.

## Forecast method and limits

The API fits an independent nonnegative seasonal ridge regression forecast to each store–product series, using up to three years of history with a linear trend and Fourier features for weekly/annual seasonality. All 500 series are fitted together in one matrix solve at startup; requests reuse their predictions. Aggregates are sums of the same per-series forecasts.

The last 30 days are held out chronologically for MAE and WAPE; this evaluation model never sees those targets. The serving model then fits all history. The indicative range uses the 90th percentile of absolute aggregate daily holdout errors for the selected series. This is a descriptive historical error range, **not a calibrated confidence interval**. Validation uses one holdout; no accuracy guarantee is implied.

CSV requirements: `date,store,item,sales`, positive integral store/item IDs, finite nonnegative sales, no duplicate series/date rows, and at least 400 complete daily dates per series. Dates missing for any series fail startup rather than being silently imputed as zero. Data is loaded into memory once per API worker. Product IDs map to the source `item` column. Store and product names in the dashboard are illustrative demo labels, not verified identities from the source dataset.

## Demo store and product names

`data/catalog.json` maps all 10 store IDs to Kerala store locations and all 50 product IDs to grocery/household names with pack sizes and categories. These names are fictional presentation labels; the source CSV has no actual store locations or product descriptions. Edit the catalog to substitute verified names, then restart the API and click **Refresh data** in the dashboard.

Names appear in the sidebar filters, store/product chart labels, planning table, and both CSV exports. Numeric IDs remain stable for API filtering and forecasting, and are retained in exports. Metadata provides `store_catalog`, `product_catalog` and `name_source`; breakdown and series records include `store_name` / `product_name`. Unknown IDs in a replacement dataset receive generic ID labels.

The dataset has **no stock-on-hand, prices, lead times or stockout records**. The app therefore reports demand forecasts, not inventory balances, revenue, reorder quantities or stockout alerts. `test.csv` contains future dates without labels; it is not used for accuracy evaluation.

The original `index.py` RNN training experiment is preserved and runs independently. The dashboard currently uses seasonal ridge regression, not its unsaved model. TensorFlow is not required for the app; install the original script's ML dependencies separately if running that experiment.

## Verify

```bash
.venv/bin/python -m pytest -q
```

Tests cover filter validation, date scope, aggregation consistency, export contents and holdout isolation. The dashboard can also be exercised with Streamlit's `AppTest` while the API is running.
