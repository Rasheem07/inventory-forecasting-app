"""Standalone Streamlit dashboard or FastAPI client via INVENTORY_API_URL."""
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests
import streamlit as st

st.set_page_config(page_title="Stockroom | Demand planning", page_icon="◈", layout="wide")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
ASSETS = Path(__file__).parent / "assets"
API = os.getenv("INVENTORY_API_URL", "").rstrip("/")
st.html(ASSETS / "tailwind.css")
st.html("""<style>
.block-container {max-width:1480px;padding-top:2.4rem;padding-bottom:3rem}
h1,h2,h3 {letter-spacing:-.035em;color:#182b44}
h1 {font-size:2.8rem !important;font-weight:750 !important}
[data-testid="stSidebar"] {border-right:1px solid #dce6ef}
[data-testid="stMetric"] {background:#fff;border:1px solid #dce6ef;border-radius:12px;padding:20px}
[data-testid="stMetricLabel"] {color:#61748a;font-size:.85rem}
[data-testid="stMetricValue"] {font-variant-numeric:tabular-nums;font-size:2rem}
[data-testid="stVerticalBlockBorderWrapper"]>div {border-color:#dce6ef !important;border-radius:14px}
.stButton>button,.stDownloadButton>button {border-radius:8px}
:focus-visible {outline:3px solid #23a6be !important;outline-offset:2px}
@media(max-width:700px){.block-container{padding:1rem}h1{font-size:2rem !important}}
@media(prefers-reduced-motion:reduce){*{animation:none !important;transition:none !important}}
</style>""")


@st.cache_resource(show_spinner="Preparing demand forecasts…")
def local_service():
    from backend.service import InventoryService
    return InventoryService(Path(os.getenv("INVENTORY_DATA_PATH", ROOT / "train.csv")))


@st.cache_data(ttl=300, show_spinner=False)
def get_json(path, params=None):
    if not API:
        service = local_service()
        if path == "/api/metadata":
            return service.metadata()
        if path == "/api/dashboard":
            pairs = params or ()
            options = dict(pairs)
            stores = [int(value) for key, value in pairs if key == "stores"]
            products = [int(value) for key, value in pairs if key == "products"]
            return service.dashboard(stores, products, options.get("start_date"),
                                     options.get("end_date"), int(options.get("horizon", 30)))
        raise ValueError("Unsupported dashboard resource")
    response = requests.get(f"{API}{path}", params=params, timeout=60)
    response.raise_for_status()
    return response.json()


def chart_layout(fig, height=330):
    fig.update_layout(template="plotly_white", height=height, margin=dict(l=10, r=10, t=20, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Arial, sans-serif", color="#61748a", size=12),
                      legend=dict(orientation="h", y=1.13, x=0), hovermode="x unified")
    fig.update_xaxes(showgrid=False, title=None)
    fig.update_yaxes(gridcolor="#eaf0f5", zeroline=False, title="Units")
    return fig


def line(fig, x, y, name, color, **kwargs):
    fig.add_trace(go.Scatter(x=x, y=y, name=name, mode="lines", line=dict(color=color, width=2.5, **kwargs)))


def number(value):
    return f"{value:,.0f}"


try:
    meta = get_json("/api/metadata")
except (requests.RequestException, ValueError):
    st.title("Stockroom")
    st.error("The forecasting API is unavailable. Start both services with `.venv/bin/python run.py`, then retry.")
    if st.button("Retry connection"):
        st.cache_data.clear()
        st.rerun()
    st.stop()

store_names = {entry["id"]: entry["name"] for entry in meta["store_catalog"]}
product_names = {entry["id"]: entry["name"] for entry in meta["product_catalog"]}

with st.sidebar:
    st.html('<div class="text-2xl font-bold text-ink mb-2">◈ &nbsp; Stockroom</div><div class="text-xs uppercase tracking-widest text-muted mb-8">Demand planning workspace</div>')
    st.subheader("Explore your inventory")
    st.caption("Empty selections include all stores or products.")
    stores = st.multiselect("Stores", meta["stores"], key="store_filter", placeholder="All stores", format_func=lambda x: store_names[x])
    products = st.multiselect("Products", meta["products"], placeholder="All products", format_func=lambda x: product_names[x])
    st.divider()
    end = date.fromisoformat(meta["history_end"])
    beginning = date.fromisoformat(meta["history_start"])
    period = st.selectbox("Sales history", ["Last 90 days", "Last 30 days", "Last year", "All history", "Custom dates"])
    days = {"Last 90 days": 90, "Last 30 days": 30, "Last year": 365}
    start = beginning if period == "All history" else max(beginning, end - timedelta(days=days.get(period, 90)-1))
    if period == "Custom dates":
        dates = st.date_input("Date range", (start, end), min_value=beginning, max_value=end)
        if len(dates) != 2:
            st.info("Select a start and end date.")
            st.stop()
        start, end = dates
    horizon = st.select_slider("Forecast horizon", options=[7, 14, 30, 60, 90], value=30, format_func=lambda x: f"{x} days")
    st.divider()
    st.caption(f"DATA THROUGH {meta['history_end']}")
    st.caption(f"{meta['rows']:,} daily observations · {meta['series']} store–product series")
    if st.button("Refresh data", width="stretch"):
        st.cache_data.clear()
        st.rerun()

params = [("stores", x) for x in stores] + [("products", x) for x in products]
params += [("start_date", start.isoformat()), ("end_date", end.isoformat()), ("horizon", horizon)]
try:
    with st.spinner("Calculating demand outlook…"):
        data = get_json("/api/dashboard", tuple(params))
except (requests.RequestException, ValueError) as exc:
    st.error("Could not load this view. Check the API connection and selected filters.")
    st.stop()

st.html('<div class="text-xs font-semibold tracking-widest uppercase text-ocean mb-2">Operations / Demand overview</div>')
left, right = st.columns([4, 1])
with left:
    st.title("A clearer view of what’s next.")
    st.caption(f"{len(stores) if stores else len(meta['stores'])} stores · {len(products) if products else len(meta['products'])} products · Sales from {start:%d %b %Y} to {end:%d %b %Y}")
with right:
    st.html(f'<div class="bg-mist rounded-lg p-4 mt-4 text-ocean text-sm font-semibold">◈ &nbsp; {horizon}-day outlook<br><span class="text-xs font-normal">From {meta["forecast_start"]}</span></div>')

summary = data["summary"]
a, b, c, d = st.columns(4)
a.metric("Units sold · selected period", number(summary["historical_units"]))
b.metric("Average daily demand", number(summary["daily_average"]))
c.metric(f"Predicted units · next {horizon} days", number(summary["forecast_units"]))
change = summary["change_vs_last_30_pct"]
d.metric("Next 30 days vs previous 30", f"{change:+.1f}%" if change is not None else "N/A", help="Forecast total for the next 30 days compared with actual sales in the final 30 days of available data, independent of the history filter.")
st.write("")

history = pd.DataFrame(data["history"])
forecast = pd.DataFrame(data["forecast"])
# Join the display traces at the actual/forecast boundary without modifying predictions.
if pd.Timestamp(forecast.iloc[0]["date"]) - pd.Timestamp(history.iloc[-1]["date"]) == pd.Timedelta(days=1):
    anchor = history.iloc[-1]
    forecast = pd.concat([pd.DataFrame([{
        "date": anchor["date"], "predicted": anchor["sales"],
        "lower": anchor["sales"], "upper": anchor["sales"],
    }]), forecast], ignore_index=True)
with st.container(border=True):
    title, controls = st.columns([3, 1])
    with title:
        st.subheader("Demand trajectory")
        st.caption("Historical sales and future demand · hover to inspect daily units")
    with controls:
        show_band = st.toggle("Show indicative range", value=True)
    fig = go.Figure()
    line(fig, history.date, history.sales, "Actual sales", "#126d82")
    if show_band:
        fig.add_trace(go.Scatter(x=forecast.date, y=forecast.upper, mode="lines", line_width=0, showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=forecast.date, y=forecast.lower, mode="lines", line_width=0, fill="tonexty", fillcolor="rgba(56,152,201,.14)", name="Indicative range", hoverinfo="skip"))
    line(fig, forecast.date, forecast.predicted, "Forecast", "#4c80d5", dash="dash")
    st.plotly_chart(chart_layout(fig, 370), width="stretch")
    st.caption(f"Forecast starts {meta['forecast_start']}, immediately after the dataset ends. {data['validation']['band_note']}")

def clear_store_filter():
    st.session_state["store_filter"] = []


left, right = st.columns([1, 1])
with left, st.container(border=True):
    st.subheader("Demand by store")
    count = len(data["by_store"])
    st.caption(f"{count} of {len(meta['stores'])} stores · Predicted units over the next {horizon} days")
    if stores:
        st.button("Show all stores", on_click=clear_store_filter,
                  help="Clear the store filter across the dashboard. Product and date filters stay selected.")
    frame = pd.DataFrame(data["by_store"])
    frame["Store"] = frame["store_name"]
    fig = px.bar(frame.sort_values("forecast_units"), x="forecast_units", y="Store", orientation="h", color_discrete_sequence=["#126d82"], labels={"forecast_units": "Predicted units"})
    fig.update_traces(width=0.65, hovertemplate="%{y}<br>%{x:,.0f} units<extra></extra>")
    fig = chart_layout(fig)
    fig.update_yaxes(title=None, showgrid=False)
    fig.update_xaxes(title="Predicted units", showgrid=True, gridcolor="#eaf0f5")
    st.plotly_chart(fig, width="stretch")
with right, st.container(border=True):
    st.subheader("Products driving demand")
    st.caption("Top 10 selected products by predicted units")
    frame = pd.DataFrame(data["by_product"]).head(10)
    frame["Product"] = frame["product_name"]
    fig = px.bar(frame.sort_values("forecast_units"), x="forecast_units", y="Product", orientation="h", color_discrete_sequence=["#4c80d5"], labels={"forecast_units": "Predicted units"})
    fig = chart_layout(fig)
    fig.update_yaxes(title=None, showgrid=False)
    fig.update_xaxes(title="Predicted units", showgrid=True, gridcolor="#eaf0f5")
    st.plotly_chart(fig, width="stretch")

left, right = st.columns([1, 1.7])
with left, st.container(border=True):
    st.subheader("The weekly rhythm")
    st.caption("Average daily sales across the selected history")
    weekly = pd.DataFrame(data["weekday"])
    fig = px.bar(weekly, x="weekday", y="average_sales", color_discrete_sequence=["#58a6a6"], labels={"average_sales": "Average units"})
    st.plotly_chart(chart_layout(fig, 260), width="stretch")
with right, st.container(border=True):
    st.subheader("Forecast reality check")
    validation = data["validation"]
    st.caption(f"Unseen 30-day holdout · {validation['start']} to {validation['end']}")
    a, b = st.columns(2)
    a.metric("Mean absolute error / series / day", f"{validation['mae']:,.2f} units")
    b.metric("Weighted absolute percentage error", f"{validation['wape_pct']:.2f}%" if validation['wape_pct'] is not None else "N/A")
    held = pd.DataFrame(validation["daily"])
    fig = go.Figure()
    line(fig, held.date, held.actual, "Actual", "#126d82")
    line(fig, held.date, held.predicted, "Held-out prediction", "#4c80d5", dash="dot")
    st.plotly_chart(chart_layout(fig, 190), width="stretch")

with st.container(border=True):
    title, download = st.columns([3, 1])
    with title:
        st.subheader("Store × product planning table")
        st.caption("Sort any column to find the fastest-growing demand or largest unit requirements.")
    rows = pd.DataFrame(data["series"]).sort_values("forecast_units", ascending=False)
    display = rows.rename(columns={"store_name": "Store", "product_name": "Product", "store": "Store ID", "product": "Product ID", "historical_units": "Historical units", "forecast_units": "Predicted units", "daily_forecast": "Daily forecast", "last_30_units": "Previous 30 days", "change_pct": "Daily demand change (%)"})
    display = display[["Store", "Product", "Historical units", "Predicted units", "Daily forecast", "Previous 30 days", "Daily demand change (%)", "Store ID", "Product ID"]]
    with download:
        st.download_button("↓ Download planning table", display.to_csv(index=False), "planning-table.csv", "text/csv", width="stretch")
    st.dataframe(display, hide_index=True, width="stretch", height=360)
    st.caption("Daily demand change compares predicted daily average with the final 30 days of actual sales.")
    if st.button("Prepare daily forecast CSV"):
        try:
            if API:
                response = requests.get(f"{API}/api/forecast.csv", params=params, timeout=60)
                response.raise_for_status()
                csv = response.content
            else:
                csv = local_service().forecast_csv(data)
            st.download_button("↓ Download daily store-product forecasts", csv, "daily-forecast.csv", "text/csv")
        except requests.RequestException:
            st.error("Export failed. Check the API connection and retry.")

with st.expander("About the data and forecasting method"):
    st.write(meta["model"])
    st.write("Each store-product model uses up to three years of history, a linear trend, and Fourier features for weekly and annual seasonality. Accuracy is measured on the final 30 days using a model fitted only on earlier observations; the live model then uses all available history.")
    st.write("The source contains sales units, store IDs and item IDs. Store locations and grocery product names are illustrative demo labels mapped to those IDs; they are not verified identities from the dataset. Stock-on-hand, prices, supplier lead times and stockout history are unavailable, so replenishment quantities, revenue and stock alerts cannot be calculated from this data.")
    st.caption("Validation uses a single historical holdout, not rolling backtests. Future accuracy can differ, especially around promotions and demand shifts.")
st.html('<div class="text-xs text-muted mt-6 pt-4 border-t border-solid border-slate-200">STOCKROOM &nbsp; / &nbsp; Demand planning · Sales units, grounded in your data</div>')
