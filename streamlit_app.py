"""Streamlit Community Cloud entrypoint. No separate API server is required."""
from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).parent / "dashboard" / "app.py"), run_name="__main__")
