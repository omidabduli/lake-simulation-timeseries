# Changelog

## Unreleased

- The app now runs completely in the browser as a static site on GitHub Pages.
  [stlite](https://github.com/whitphx/stlite) runs the unchanged Streamlit app
  on Pyodide, so workbooks never leave the user's device and no server is
  needed.
- Every push to `main` runs the tests, builds the site
  (`scripts/build_site.py`) and deploys it (`.github/workflows/deploy-pages.yml`).
- When the `shap` package is not available (it cannot be installed in the
  browser), SHAP values come from XGBoost's built-in TreeSHAP, which returns
  the same values as `shap.TreeExplainer`.
- The training loop is async, so the live training view keeps updating in the
  browser. Results are the same as on a normal Streamlit server.
- Renamed the project to Universal Time-Series Forecasting. The lake data is
  now only the demo (`Example/Lake_Demo_2000_Rows.xlsx`).
- The Docker setup moved to `docker/`.

### Fixed

- The same workbook could give a different model on a later run on a
  Streamlit server, because the model search used Python's global random
  generator, which other parts of the server also draw from. It now uses its
  own generator with a fixed seed.
- The file upload texts use the same font in the browser as on the server.

## 2.0.0 (2026-08-01)

- New interface: clearer layout, better contrast, working sidebar on small
  screens, and clear states for loading, training, success and errors.
- Live training view that shows each candidate model, the validation progress
  and the best score so far.
- Advanced settings for search effort, number of validation folds, model
  complexity and rolling windows, each with an explanation.
- Scenario dates continue the time index of the history, so the long-term
  trend no longer restarts at zero when a scenario begins.
- Seasonal features can be switched off, and longer rolling windows (14 and 30
  observations) are available.
- Demo workbook with 2,000 days of history and a 365-day warming scenario.
- Docker Compose setup with persistent logs, automatic restart and a health
  check.
- Tests for the continuous time index and the rolling windows.
