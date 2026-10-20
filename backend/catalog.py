"""Editable demo labels. Numeric IDs remain the source of identity."""
import json
from pathlib import Path

CATALOG_PATH = Path(__file__).resolve().parents[1] / "data" / "catalog.json"
CATALOG = json.loads(CATALOG_PATH.read_text())
STORE_NAMES = {entry["id"]: entry["name"] for entry in CATALOG["stores"]}
PRODUCT_NAMES = {entry["id"]: entry["name"] for entry in CATALOG["products"]}
PRODUCT_CATEGORIES = {entry["id"]: entry["category"] for entry in CATALOG["products"]}


def store_name(identifier):
    return STORE_NAMES.get(int(identifier), f"Store {identifier:02d}")


def product_name(identifier):
    return PRODUCT_NAMES.get(int(identifier), f"Product {identifier:02d}")
