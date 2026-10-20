"""Daily demand forecasts; no stock or price assumptions are made."""
from pathlib import Path

import numpy as np
import pandas as pd

from backend.catalog import CATALOG, PRODUCT_CATEGORIES, product_name, store_name


class InventoryService:
    model_name = "Seasonal ridge regression · annual + weekly seasonality + trend"

    def __init__(self, path: Path):
        data = pd.read_csv(path, parse_dates=["date"])
        required = {"date", "store", "item", "sales"}
        if not required.issubset(data.columns):
            raise ValueError(f"Dataset must contain {sorted(required)}")
        if data[list(required)].isna().any().any():
            raise ValueError("Dataset contains missing values")
        for column in ("store", "item", "sales"):
            data[column] = pd.to_numeric(data[column], errors="raise")
            if not np.isfinite(data[column]).all() or (data[column] < 0).any():
                raise ValueError(f"Invalid {column} values")
        for column in ("store", "item"):
            if (data[column] % 1 != 0).any() or (data[column] < 1).any():
                raise ValueError("Store and item IDs must be positive integers")
        if data.duplicated(["date", "store", "item"]).any():
            raise ValueError("Duplicate store-item dates")
        self.panel = data.pivot(index="date", columns=["store", "item"], values="sales").sort_index()
        dates = pd.date_range(self.panel.index.min(), self.panel.index.max())
        self.panel = self.panel.reindex(dates)
        if self.panel.isna().any().any() or len(dates) < 400:
            raise ValueError("At least 400 complete daily observations per store-item are required")
        self.dates = dates
        self.values = self.panel.to_numpy(dtype=float)
        self.stores = sorted(int(x) for x in data.store.unique())
        self.products = sorted(int(x) for x in data.item.unique())
        self.pairs = self.panel.columns
        # Freeze a genuine final-30-day holdout before fitting the production model.
        self.validation_actual = self.values[-30:]
        self.validation_predicted = self._predict(self.values[:-30], self.dates[-30:])
        self.forecast_dates = pd.date_range(dates[-1] + pd.Timedelta(days=1), periods=90)
        self.predictions = self._predict(self.values, self.forecast_dates)

    def _features(self, dates):
        t = np.asarray((pd.DatetimeIndex(dates) - self.dates[0]).days, dtype=float)
        features = [np.ones(len(t)), t / 365.25]
        for period, harmonics in ((7, 3), (365.25, 6)):
            for k in range(1, harmonics + 1):
                features += [np.sin(2 * np.pi * k * t / period), np.cos(2 * np.pi * k * t / period)]
        return np.column_stack(features)

    def _predict(self, values, future_dates):
        # Recent three years balance seasonal coverage and responsiveness.
        start = max(0, len(values) - 1095)
        x = self._features(self.dates[start:len(values)])
        penalty = np.eye(x.shape[1]) * 2.0
        penalty[0, 0] = 0
        coefficients = np.linalg.solve(x.T @ x + penalty, x.T @ values[start:])
        return np.maximum(0, self._features(future_dates) @ coefficients)

    def metadata(self):
        return {
            "stores": self.stores, "products": self.products,
            "store_catalog": [{"id": i, "name": store_name(i)} for i in self.stores],
            "product_catalog": [{"id": i, "name": product_name(i), "category": PRODUCT_CATEGORIES.get(i, "Other")} for i in self.products],
            "name_source": CATALOG["name_source"],
            "history_start": str(self.dates[0].date()), "history_end": str(self.dates[-1].date()),
            "forecast_start": str(self.forecast_dates[0].date()),
            "rows": int(self.values.size), "series": len(self.pairs),
            "model": self.model_name, "default_horizon": 30, "max_horizon": 90,
            "stock_available": False,
        }

    def selection(self, stores, products):
        if stores and not set(stores).issubset(self.stores):
            raise ValueError("Unknown store ID")
        if products and not set(products).issubset(self.products):
            raise ValueError("Unknown product ID")
        mask = np.array([(not stores or s in stores) and (not products or p in products)
                         for s, p in self.pairs])
        if not mask.any():
            raise ValueError("No store-product series match these filters")
        return mask

    def dashboard(self, stores=None, products=None, start=None, end=None, horizon=30):
        selected = self.selection(stores, products)
        start = pd.Timestamp(start) if start else self.dates[-1] - pd.Timedelta(days=89)
        end = pd.Timestamp(end) if end else self.dates[-1]
        if start > end:
            raise ValueError("start_date must be on or before end_date")
        window = (self.dates >= start) & (self.dates <= end)
        if not window.any():
            raise ValueError("Date range does not overlap sales history")
        actual = self.values[:, selected]
        future = self.predictions[:horizon, selected]
        validation_actual = self.validation_actual[:, selected]
        validation_predicted = self.validation_predicted[:, selected]
        residual = validation_actual - validation_predicted
        total = validation_actual.sum()
        mae = float(np.abs(residual).mean())
        wape = float(np.abs(residual).sum() / total * 100) if total else None
        # Aggregate residuals directly: do not assume independent store-item errors.
        aggregate_error = np.quantile(np.abs(residual.sum(axis=1)), .9)
        forecast_sum = future.sum(axis=1)
        history_sum = actual.sum(axis=1)
        last_30 = float(history_sum[-30:].sum())
        next_30 = float(self.predictions[:30, selected].sum())
        selected_pairs = list(self.pairs[selected])
        records = []
        for i, (store, product) in enumerate(selected_pairs):
            recent = float(actual[-30:, i].sum())
            demand = float(future[:, i].sum())
            records.append({"store": int(store), "product": int(product),
                            "store_name": store_name(store), "product_name": product_name(product),
                            "historical_units": float(actual[window, i].sum()),
                            "forecast_units": round(demand, 2),
                            "daily_forecast": round(demand / horizon, 2),
                            "last_30_units": recent,
                            "change_pct": round((future[:, i].mean() / (recent / 30) - 1) * 100, 2) if recent else None})
        frame = pd.DataFrame(records)
        def breakdown(key):
            grouped = frame.groupby(key)[["historical_units", "forecast_units"]].sum().reset_index()
            label = store_name if key == "store" else product_name
            grouped[f"{key}_name"] = grouped[key].map(label)
            return grouped.sort_values("forecast_units", ascending=False).round(2).to_dict("records")
        return {
            "filters": {"stores": stores or self.stores, "products": products or self.products,
                        "start_date": str(start.date()), "end_date": str(end.date()), "horizon": horizon},
            "summary": {"historical_units": float(actual[window].sum()),
                        "daily_average": float(actual[window].sum(axis=1).mean()),
                        "forecast_units": round(float(future.sum()), 2),
                        "next_30_units": round(next_30, 2),
                        "change_vs_last_30_pct": round((next_30 / last_30 - 1) * 100, 2) if last_30 else None,
                        "active_series": len(selected_pairs)},
            "history": [{"date": str(d.date()), "sales": float(v)}
                        for d, v in zip(self.dates[window], history_sum[window])],
            "forecast": [{"date": str(d.date()), "predicted": round(float(v), 2),
                          "lower": round(max(0, float(v - aggregate_error)), 2),
                          "upper": round(float(v + aggregate_error), 2)}
                         for d, v in zip(self.forecast_dates[:horizon], forecast_sum)],
            "by_store": breakdown("store"), "by_product": breakdown("product"),
            "series": records,
            "weekday": [{"weekday": name, "average_sales": round(float(history_sum[window][self.dates[window].dayofweek == i].mean()), 2)}
                        for i, name in enumerate(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
                        if (self.dates[window].dayofweek == i).any()],
            "validation": {"mae": round(mae, 3), "wape_pct": round(wape, 2) if wape is not None else None,
                           "start": str(self.dates[-30].date()), "end": str(self.dates[-1].date()),
                           "method": self.model_name,
                           "band_note": "Indicative range: ±90th percentile of absolute aggregate daily errors in a single 30-day holdout. Not a calibrated confidence interval.",
                           "daily": [{"date": str(d.date()), "actual": float(a), "predicted": round(float(p), 2)}
                                     for d, a, p in zip(self.dates[-30:], validation_actual.sum(axis=1), validation_predicted.sum(axis=1))]},
        }

    def forecast_csv(self, data):
        """Export the same per-series predictions in API and standalone modes."""
        mask = self.selection(data["filters"]["stores"], data["filters"]["products"])
        n = data["filters"]["horizon"]
        frame = pd.DataFrame(self.predictions[:n, mask], index=self.forecast_dates[:n],
                             columns=self.pairs[mask])
        frame.index.name = "date"
        records = frame.stack([0, 1], future_stack=True).rename("predicted_units").reset_index()
        records = records.rename(columns={"item": "product"})
        records.insert(2, "store_name", records["store"].map(store_name))
        records.insert(4, "product_name", records["product"].map(product_name))
        return records.round(2).to_csv(index=False)
