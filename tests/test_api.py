import io

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as client:
        yield client


def test_metadata_and_default_forecast(client):
    meta = client.get("/api/metadata").json()
    assert meta["series"] == 500
    result = client.get("/api/dashboard")
    assert result.status_code == 200
    data = result.json()
    assert len(data["forecast"]) == 30
    assert data["forecast"][0]["date"] == "2018-01-01"
    assert len(data["history"]) == 90
    for row in data["forecast"]:
        assert 0 <= row["lower"] <= row["predicted"] <= row["upper"]
    assert data["validation"]["wape_pct"] > 0


def test_filters_and_additive_forecast(client):
    one = client.get("/api/dashboard", params={"stores": 1, "products": 1}).json()
    two = client.get("/api/dashboard", params={"stores": 2, "products": 1}).json()
    both = client.get("/api/dashboard", params=[("stores", 1), ("stores", 2), ("products", 1)]).json()
    assert one["summary"]["active_series"] == 1
    assert one["series"][0]["product"] == 1
    assert both["summary"]["forecast_units"] == pytest.approx(one["summary"]["forecast_units"] + two["summary"]["forecast_units"], abs=.02)
    assert both["summary"]["historical_units"] == one["summary"]["historical_units"] + two["summary"]["historical_units"]


@pytest.mark.parametrize("params", [
    {"stores": 999}, {"products": 999}, {"stores": "abc"},
    {"horizon": 0}, {"horizon": 91}, {"start_date": "bad-date"},
    {"start_date": "2017-12-01", "end_date": "2017-01-01"},
    {"start_date": "2026-01-01"},
])
def test_rejects_invalid_filters(client, params):
    assert client.get("/api/dashboard", params=params).status_code == 422


def test_dates_filter_history_only(client):
    params = {"stores": 1, "products": 2, "start_date": "2014-01-01", "end_date": "2014-01-07"}
    data = client.get("/api/dashboard", params=params).json()
    assert len(data["history"]) == 7
    assert data["forecast"][0]["date"] == "2018-01-01"
    original = pd.read_csv("train.csv")
    expected = original[(original.store == 1) & (original.item == 2) & original.date.between("2014-01-01", "2014-01-07")].sales.sum()
    assert data["summary"]["historical_units"] == expected


def test_csv_and_endpoint_consistency(client):
    params = [("stores", 1), ("products", 1), ("products", 2), ("horizon", 7)]
    response = client.get("/api/forecast.csv", params=params)
    assert response.status_code == 200
    frame = pd.read_csv(io.StringIO(response.text))
    assert set(frame.columns) == {"date", "store", "store_name", "product", "product_name", "predicted_units"}
    assert len(frame) == 14
    assert frame.store_name.eq("Kochi · Edappally").all()
    assert set(frame.product_name) == {"Basmati rice · 1 kg", "Sona masoori rice · 1 kg"}
    dashboard = client.get("/api/dashboard", params=params).json()
    assert frame.predicted_units.sum() == pytest.approx(dashboard["summary"]["forecast_units"], abs=.1)
    assert client.get("/api/forecast", params=params).json()["forecast"] == dashboard["forecast"]
    assert client.get("/api/breakdown/store", params=params).json()["data"] == dashboard["by_store"]
    assert client.get("/api/breakdown/nope").status_code == 422


def test_holdout_does_not_leak_future(client):
    service = app.state.inventory
    baseline = service._predict(service.values[:-30], service.dates[-30:])
    changed = service.values.copy()
    changed[-30:] += 100000
    recomputed = service._predict(changed[:-30], service.dates[-30:])
    np.testing.assert_allclose(baseline, recomputed)
    np.testing.assert_allclose(baseline, service.validation_predicted)
