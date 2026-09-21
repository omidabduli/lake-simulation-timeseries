# Changelog

All notable changes to Lake Time-Series Forecasting are documented here.

## [Unreleased]

### GitHub Pages deployment

- The application now runs entirely in the browser as a static site on GitHub
  Pages. [stlite](https://github.com/whitphx/stlite) runs the unchanged
  Streamlit app on Pyodide (Python compiled to WebAssembly); workbooks never
  leave the user's device and no server is needed.
- Every push to `main` tests, builds (`scripts/build_site.py`) and deploys the
  site through `.github/workflows/deploy-pages.yml`.
- When the `shap` package is unavailable (the browser runtime cannot install
  it), SHAP values come from XGBoost's built-in TreeSHAP, which returns the
  same values as `shap.TreeExplainer`.
- The scenario pipeline is async so the live training console keeps updating
  in the browser; results are unchanged (verified bit-for-bit against the
  previous server version).

### Fixes

- Reproducible results on a Streamlit server: the hyperparameter search used
  the global `random` generator, which WebSocket keep-alive pings also draw
  from, so the same workbook could yield a different model on a later run.
  It now uses a private `random.Random(42)` with the identical sequence.
- The file uploader texts use the app font (Inter) in the browser build, as
  they do on the server.
- The Hetzner/CloudPanel deployment script and the Docker setup moved to
  `legacy/server/`; the deployment script no longer contains credentials.

## [2.0.0] — 2026-08-01

Version 2.0.0 is a major product release that introduces a complete visual
redesign, a clearer forecasting workflow, expanded model controls, and a
verified Docker deployment path.

### Complete interface redesign

- Rebuilt the application around a professional Roland Digital-inspired visual
  system with warm neutral surfaces, electric-blue accents, and consistent
  scientific typography.
- Improved contrast and readability across metrics, target labels, status
  badges, outlier values, expanders, charts, upload controls, and messages.
- Restored reliable sidebar collapse and reopen controls.
- Removed non-functional heading link icons and legacy dark-theme styling.
- Added responsive layouts for smaller screens and clearer empty, loading,
  validation, training, success, and error states.

### Better training experience

- Added an animated live-training console so users can see candidate models,
  forward-only validation progress, and the best score while optimization runs.
- Replaced generic progress feedback with scenario-specific model status and
  clearer explanations of each stage.
- Improved result cards, scorecards, feature labels, and SHAP visualizations.

### Forecasting and model-control improvements

- Added explained Advanced-mode controls for search effort, validation folds,
  model complexity, and rolling context windows.
- Preserved a continuous historical time origin when generating future scenario
  features, preventing the future trend index from restarting at zero.
- Added configurable seasonal feature engineering and extended rolling windows.
- Added focused regression tests for continuous time and rolling-window logic.

### Reliability fixes

- Corrected Streamlit uploader styling selectors and sidebar behavior.
- Fixed low-contrast text on dark badges and inline outlier values.
- Improved long target-variable and scientific feature-name presentation.
- Updated documentation and labels to match actual application behavior.
- Standardized the product name as **Lake Time-Series Forecasting** throughout
  the application and documentation.

### Demo and deployment

- Added a ready-to-run workbook with 2,000 historical rows and a future warming
  scenario for immediate evaluation.
- Added Docker Compose deployment with persistent logs and automatic restart.
- Included the production Streamlit theme inside the Docker image.
- Added a built-in container health check using Streamlit's health endpoint.
- Verified the v2.0.0 image by building it locally, starting the application,
  and confirming a healthy container state.

[2.0.0]: https://github.com/omidabduli/lake-simulation-timeseries/releases/tag/v2.0.0
