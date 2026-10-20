"""Read-only inventory demand API. Run with uvicorn backend.main:app."""
import os
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import Response

from backend.service import InventoryService

ROOT = Path(__file__).resolve().parents[1]


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.inventory = InventoryService(Path(os.getenv("INVENTORY_DATA_PATH", ROOT / "train.csv")))
    yield


app = FastAPI(title="Stockroom · Inventory Forecast API", version="1.0.0", lifespan=lifespan,
              description="Historical demand, filtered analytics and up to 90 future daily forecasts. Product IDs correspond to item in train.csv.")


def service(request: Request) -> InventoryService:
    return request.app.state.inventory


@app.get("/api/health", tags=["System"])
def health(inventory: Annotated[InventoryService, Depends(service)]):
    return {"status": "ok", "series": len(inventory.pairs)}


@app.get("/api/metadata", tags=["Dashboard"])
def metadata(inventory: Annotated[InventoryService, Depends(service)]):
    return inventory.metadata()


def filtered_dashboard(
    inventory: Annotated[InventoryService, Depends(service)],
    stores: Annotated[list[int] | None, Query(max_length=100, description="Repeat: stores=1&stores=2")] = None,
    products: Annotated[list[int] | None, Query(max_length=500, description="Product IDs (item column)")] = None,
    start_date: date | None = None,
    end_date: date | None = None,
    horizon: Annotated[int, Query(ge=1, le=90)] = 30,
):
    try:
        return inventory.dashboard(stores, products, start_date, end_date, horizon)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


Dashboard = Annotated[dict, Depends(filtered_dashboard)]


@app.get("/api/dashboard", tags=["Dashboard"])
def dashboard(data: Dashboard):
    """All dashboard charts and metrics with a consistent set of filters."""
    return data


@app.get("/api/history", tags=["Analytics"])
def history(data: Dashboard):
    return {"filters": data["filters"], "history": data["history"]}


@app.get("/api/forecast", tags=["Analytics"])
def forecast(data: Dashboard):
    return {"filters": data["filters"], "forecast": data["forecast"], "validation": data["validation"]}


@app.get("/api/breakdown/{dimension}", tags=["Analytics"])
def breakdown(dimension: str, data: Dashboard):
    if dimension not in {"store", "product"}:
        raise HTTPException(422, "dimension must be store or product")
    return {"filters": data["filters"], "data": data[f"by_{dimension}"]}


@app.get("/api/forecast.csv", tags=["Export"])
def export(data: Dashboard, inventory: Annotated[InventoryService, Depends(service)]):
    """Daily forecasts for each selected store-product pair."""
    return Response(inventory.forecast_csv(data), media_type="text/csv",
                    headers={"Content-Disposition": 'attachment; filename="inventory-forecast.csv"'})
