"""Browser entrypoint for the GitHub Pages build (stlite on Pyodide).

stlite starts this file instead of app.py. It runs app.py unchanged and then
awaits the scenario coroutine that app.py leaves in `scenario_run` when it
detects the browser runtime: Pyodide's event loop is already running, so
app.py cannot call asyncio.run() itself. Awaiting here (stlite supports
top-level await) also lets the live training console repaint while models train.
"""
import runpy
from pathlib import Path

page = runpy.run_path(str(Path(__file__).with_name("app.py")), run_name="__main__")
await page["scenario_run"]
