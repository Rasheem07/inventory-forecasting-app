"""Start the API and dashboard together, and stop both on exit."""
import os
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent


def main():
    api_port = int(os.getenv("API_PORT", "8000"))
    ui_port = int(os.getenv("DASHBOARD_PORT", "8501"))
    env = dict(os.environ, INVENTORY_API_URL=f"http://127.0.0.1:{api_port}")
    processes = []
    try:
        api = subprocess.Popen([sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", str(api_port)], cwd=ROOT, env=env)
        processes.append(api)
        for _ in range(120):
            if api.poll() is not None:
                raise RuntimeError("API exited during startup. Check logs above.")
            try:
                response = requests.get(f"{env['INVENTORY_API_URL']}/api/health", timeout=1)
                if response.ok:
                    break
            except requests.RequestException:
                pass
            time.sleep(.5)
        else:
            raise RuntimeError("API did not become ready within 60 seconds")
        ui = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "dashboard/app.py", "--server.address", "127.0.0.1", "--server.port", str(ui_port)], cwd=ROOT, env=env)
        processes.append(ui)
        print(f"\nDashboard: http://127.0.0.1:{ui_port}\nAPI docs: http://127.0.0.1:{api_port}/docs\n", flush=True)
        while all(p.poll() is None for p in processes):
            time.sleep(.5)
    except KeyboardInterrupt:
        pass
    finally:
        for process in reversed(processes):
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == "__main__":
    main()
