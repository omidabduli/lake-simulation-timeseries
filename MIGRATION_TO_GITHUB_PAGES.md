# Migration to GitHub Pages

Date: 2026-09-21 · Repository: `omidabduli/lake-simulation-timeseries`

**Result: the application runs fully on GitHub Pages. No server is required.**
Target URL: <https://omidabduli.github.io/lake-simulation-timeseries/> (later:
`https://time-series.roland-digital.de`).

## Classification

**Category B – mostly client-side.** The Hetzner server only performed
computations on the workbook that each user uploads during their session:
Excel parsing, data-quality checks, feature engineering, XGBoost training with
time-series cross-validation, SHAP explanations, Plotly charts and a CSV
export. There is no database, no authentication, no secret used by the
application, no external API, no scheduled job, and no data shared between
users. The only server-side side effect is a JSON run log in `logs/`, which no
feature reads back.

All of this runs in the browser. Instead of translating the Python algorithms
to JavaScript (which would have meant re-implementing XGBoost and TreeSHAP and
would have changed the numbers), the same Python application runs in the
browser, with two small adaptations (section 5), using [stlite](https://github.com/whitphx/stlite): Streamlit on
[Pyodide](https://pyodide.org), CPython compiled to WebAssembly. After the
migration the project is a **fully static site (category A)**.

---

## 1. Original architecture

```text
Browser ──WebSocket──► nginx (CloudPanel) ──► Streamlit server (Python) on the Hetzner server
                                              app.py + modules/ (pandas, XGBoost, SHAP, Plotly)
```

- **Application:** Streamlit app (`app.py`, `modules/`). The browser was a thin
  client; uploaded workbooks were sent to the server and all computation ran there.
- **Deployment:** `deploy.command`, run on the Mac. It logged in to the server
  with a password (`sshpass`, host-key checking disabled), `rsync`ed the
  repository to `/home/<SITE_USER>/htdocs/<SITE_DOMAIN>`, created a virtual
  environment, installed `requirements.txt`, killed the previous process and
  started `nohup streamlit run app.py --server.port=8091 --server.address=0.0.0.0`
  with CORS and XSRF protection disabled. CloudPanel's nginx proxied
  `https://time-series.roland-digital.de` to that port. There was no systemd
  service, so the app did not restart after a reboot.
- **Configuration:** `.env` (server IP, SSH user, password, domain, app port,
  SSH port). The same values were also hard-coded as fallbacks in `deploy.command`.
- **DNS:** `time-series.roland-digital.de` → `A` record pointing at the Hetzner
  server; the zone is hosted at Hetzner (`ns1.your-server.de`).
- **Also present:** `Dockerfile` and `docker-compose.yml` (generic container on
  port 8501, not used by the Hetzner deployment) and `run.command` (local
  launcher on port 8502).
- **Status observed on 2026-09-21:** the domain still resolves to the server,
  but no site answers (HTTP returns an empty reply; HTTPS fails with the TLS
  alert `unrecognized_name`), so the Hetzner deployment was not serving the app.

## 2. New architecture

```text
Browser ──HTTPS──► GitHub Pages: index.html, app/manifest.json, app/*.py, demo workbook
   │
   ├─► jsDelivr: stlite 1.9.1 (Streamlit 1.62) and Pyodide 0.29.3 + numpy, pandas, scipy,
   │             scikit-learn, xgboost, xlrd
   ├─► PyPI:     openpyxl 3.1.5, plotly 6.9.0 (pure-Python wheels)
   └─► Web Worker: Pyodide runs app.py → pandas / scikit-learn / XGBoost / Plotly
                   → charts, SHAP, CSV download, all on the device
```

- `web/index.html` loads stlite, reads `app/manifest.json` and mounts the Python
  files into the browser's virtual file system. A start-up screen in the app's
  design is shown while Python boots.
- `web/entrypoint.py` is the browser entrypoint. It runs `app.py` unchanged and
  awaits its scenario pipeline (see section 5).
- `scripts/build_site.py` assembles `_site/` using only the standard library.
- `.github/workflows/deploy-pages.yml` tests, builds and deploys on every push to `main`.
- Privacy improvement: uploaded workbooks are no longer transmitted anywhere.

## 3. What was changed

| File | Change |
| --- | --- |
| `app.py` | `_run_scenario` and the run/render section are coroutines; `time.sleep(0.15)` became `await asyncio.sleep(0.15)`; three zero-length yields (0.05 s in the browser) let the browser paint the run header, live console and "Finalizing" step. A Streamlit server runs the coroutine with `asyncio.run()`. `IS_BROWSER` detects Pyodide. No calculation was touched. |
| `app.py` (bug fix) | The hyperparameter search uses a private `random.Random(42)` instead of `random.seed(42)` on the shared global generator. It is the same sequence, but on a server other threads (WebSocket keep-alive pings) could shift it, so the same workbook sometimes gave different results. See section 5. |
| `app.py` (CSS) | The global font rule also targets `.stlite-root` and `[data-st-overlay-root]`, where stlite applies the theme font instead of `<body>`. On a Streamlit server these rules change nothing. |
| `web/index.html` | Runs the browser build with `client.toolbarMode = viewer` (as on a public server) and `server.fileWatcherType = none` (published files never change). |
| `modules/explainer.py` | `shap` is imported optionally. Without it (browser), TreeSHAP values come from XGBoost's built-in `pred_contribs`, the same call `shap.TreeExplainer` makes internally for XGBoost models. |
| `modules/docs_view.py` | In-app documentation: "08. Deployment" now describes GitHub Pages, and Docker points to `legacy/server/`; the architecture tree lists `web/` and `scripts/`. |
| `README.md`, `CHANGELOG.md` | Local development, production build, deployment, URL, custom domain, architecture, remaining dependencies. |
| `legacy/server/*` | Docker files moved here (paths adjusted, `.env` and `.venv` now excluded from the build context). |

**Created:** `web/index.html`, `web/entrypoint.py`, `web/requirements.txt`,
`web/404.html`, `web/favicon.png` (Streamlit's default icon, as before),
`scripts/build_site.py`, `.github/workflows/deploy-pages.yml`, `.gitignore`,
`tests/test_explainer.py`, `tests/test_build_site.py`,
`tests/test_hyperparameter_search.py`, `legacy/server/README.md`,
`legacy/server/.env.example`, this report.

**Local only (not committed):** `legacy/server/deploy.command` (sanitized),
entries in `.git/info/exclude` (`design.md`, `.design_ref/`, `deploy.command`,
`CLAUDE.md`, `.claude/`, `tests/test_docs_view.py`), which keeps these private
files ignored without listing their names in the public `.gitignore`.

## 4. What was removed

- **Nothing was removed from the application.** All features and the interface
  are preserved; only the in-app "Deployment" documentation text was updated.
- `deploy.command` (repository root) contained the server password in plain
  text. It was replaced by `legacy/server/deploy.command`, which reads every
  value from `.env` and has no fallbacks. The original file was moved to the
  macOS **Trash** as `deploy.command (original with hardcoded credentials)`.
  Its hard-coded values were identical to `.env`, so nothing is lost. Empty the
  Trash once you no longer need it.
- `Dockerfile`, `docker-compose.yml` and `.dockerignore` were moved (not
  deleted) to `legacy/server/`.

## 5. What was converted from backend to frontend

Everything: Excel import (openpyxl/xlrd), sheet separation, time-index
detection, target detection, placeholder/IQR audit and cleaning, cyclical and
rolling features, the random hyperparameter search over `TimeSeriesSplit`,
the final fit, predictions, TreeSHAP, Plotly charts, the German-locale CSV
export and the demo-file download. The Python code is the same code.

Two adaptations were needed for the browser runtime:

1. **SHAP:** `shap` cannot be installed in Pyodide (it depends on numba, a JIT
   compiler that cannot run in WebAssembly). For XGBoost models,
   `shap.TreeExplainer(model).shap_values(X)` is exactly
   `booster.predict(DMatrix(X), pred_contribs=True)[:, :-1]`, and the
   fallback uses that call. Verified bit-identical (max. difference 0.0).
2. **Live progress:** in the browser, Python shares one thread with the UI
   message loop, so updates are painted only when the script yields.
   `time.sleep()` is a no-op there. The pipeline now `await`s at the points
   where it previously slept, so the training console still animates.

### Verification: old vs. new implementation

The real `_run_scenario` from `app.py` was run on three inputs (the demo
workbook with default settings; demo with seasonal features off, 8
candidates, 3 folds, windows 3/7/14/30; a synthetic hourly workbook with gaps
and `-999` placeholders, 5 folds, depth up to 10):

| Comparison | R², RMSE | Best hyperparameters | 365 predictions | Mean \|SHAP\| | CSV, data-quality flags |
| --- | --- | --- | --- | --- | --- |
| Original code vs. migrated code, both on a Streamlit server | identical (10 decimals) | identical | identical (max. diff 0) | identical | identical |
| Original server version (Python 3.11, pandas 3.0.5, XGBoost 3.2.0, shap) vs. browser runtime (Pyodide 0.29.3: Python 3.13 wasm32, pandas 2.3.3, XGBoost 2.1.4, XGBoost TreeSHAP) | identical | identical | identical (max. diff 0) | identical | identical |

End-to-end in the browser, the demo workbook gave R² 0.9843, RMSE 0.18759,
36 factors and the same top-10 SHAP bars and predictions as the server
version. The downloaded CSV matched byte-for-byte in its header and rows.
Training took about 20 s in the browser versus about 19–22 s on the server
(single-threaded WebAssembly is competitive here).

**Visual parity:** the computed font family, size, weight and colour of every
visible text were compared between the Streamlit server and the GitHub Pages
build. That covered 303 texts on the home, review and results pages and 664
texts across all 9 documentation tabs, and found **0 differences** (after the
font fix above). Screenshots at desktop (1280×900, 800×600) and phone
(390×844, touch) sizes are identical apart from the "Deploy" button, which
Streamlit shows only on `localhost`.

Other checks: start-up screen and boot (~10 s with a warm cache), sidebar,
interactive hero, documentation view and back, file upload, data-quality
review, "Proceed as provided", live training console (all 15 candidates
repaint), result tabs, metrics, both Plotly charts, CSV and example
downloads, a 404 page, sub-path hosting (the site was also served from
`/__test/a/` sub-directories), no `localhost` or server addresses in the
published files, and 10 unit tests plus the build on a clean checkout with
Python 3.13 (as in CI).

### Reproducibility bug found in the server version (fixed)

On a real Streamlit server, consecutive runs of the same workbook did not
always produce the same model. The original code gave R² 0.9843, 0.9843,
then 0.9839. `_run_scenario` seeded the **global** `random` module and drew
hyperparameters from it during a ~20 s search. Streamlit's server
(uvicorn/websockets) sends WebSocket keep-alive pings with a payload from
`random.getrandbits(32)` on that same global generator, and any ping during the
search shifted all later candidates.

The search now uses a private `random.Random(42)`, which yields exactly the
sequence `random.seed(42)` did (verified for 30 candidates), so the correct
results are unchanged. With simulated pings every 1.3 s the result stays at
R² 0.9842880312, and three consecutive server runs now all give 0.9843.
`tests/test_hyperparameter_search.py` fails on the old code and passes on the
new. The browser build was not affected in practice (it has no WebSocket
pings), but it benefits from the same guarantee.

### Stability testing

During manual testing in the desktop app's embedded browser, the Python
runtime in one tab crashed once ("Pyodide has suffered a fatal error: table
index is out of bounds", raised inside Streamlit's polling file watcher) a
few seconds after the second CSV download of that session. That session had
been heavily instrumented (the tests overrode browser download functions). It
could not be reproduced in four independent headless Chrome 153 sessions,
each with three full training runs, reruns, CSV and example downloads, and
2.5–4 minutes idle, with the file watcher on and off. Every session stayed
responsive.

Because published files never change, the browser build now disables
Streamlit's file watcher, which polls several times per second. This removes
the code path where the fault appeared and saves CPU and battery for
visitors. If a tab ever stops responding, reloading the page starts a fresh
Python runtime.

## 6. Remaining backend dependencies

**None for the application.** At runtime the browser loads versioned,
public, cacheable files from jsDelivr (stlite, Pyodide and its packages),
PyPI (openpyxl, plotly) and Google Fonts (typography, as before). No
credentials are involved.

Not available without a server: **central collection of the JSON run logs**.
`save_simulation_log()` still runs, but writes into the tab's temporary
in-memory file system, so each visitor's logs disappear when the tab closes.
If you want these logs centrally, it would need a small serverless endpoint
(for example a Cloudflare Worker) that the page posts to. That would also
mean publishing a privacy notice, because run metadata would then leave the
user's device.

## 7. Security issues found

1. **Server password in the public Git history (critical).**
   `deploy.command` with the Hetzner/CloudPanel **server IP, SSH user,
   password, domain and app port** hard-coded was committed in `4f1e46c`
   (2026-08-07 12:47) and deleted three minutes later in `48e6ab0`. Deleting
   a file does not remove it from history, and both commits are on
   `origin/main` of the **public** repository `omidabduli/lake-simulation-timeseries`
   (the `noybiss/…` remote redirects to it). Treat the password as compromised:
   - **Change the password now** (the CloudPanel site user, and anywhere else it is reused).
   - Prefer SSH-key authentication and disable password logins for that user.
   - Optionally rewrite history (`git filter-repo --path deploy.command --invert-paths`)
     and force-push. That rewrites every commit hash, and forks or clones may
     still contain the file, so rotating the password is the essential step.
2. `.env` (same credentials, plain text) exists only locally. It was never
   committed (verified across all history) and is ignored by `.gitignore`.
3. The old deploy script disabled SSH host-key verification
   (`StrictHostKeyChecking=no`, man-in-the-middle risk) and used password
   logins. The sanitized legacy copy keeps its behaviour but is documented as legacy.
4. The server ran Streamlit with `enableXsrfProtection=false` and
   `enableCORS=false`. This is irrelevant for the static site.
5. The old `.dockerignore` did not exclude `.env`, so it was sent to the Docker
   build context (not copied into the image). Fixed.
6. A full scan of the working tree and **every blob in the Git history** for
   private keys, GitHub/AWS/Google/Slack/OpenAI-style tokens, password
   assignments and the `.env` values found **nothing else**. The published site
   contains no secrets, and the app needs none.

## 8. GitHub Pages deployment configuration

- **Workflow:** `.github/workflows/deploy-pages.yml`, triggered by `push` to
  `main` and by manual *Run workflow*. Permissions: `contents: read`,
  `pages: write`, `id-token: write`. Concurrency group `pages` (no
  cancellation of a running deployment).
- **Build job:** `actions/checkout@v7` → `actions/setup-python@v7` (Python
  3.13, pip cache) → `pip install -r requirements.txt` → `python -m unittest
  discover -s tests` → `python scripts/build_site.py --out _site` →
  `actions/configure-pages@v6` → `actions/upload-pages-artifact@v5`.
- **Deploy job:** `actions/deploy-pages@v5` into the `github-pages` environment.
- **Site layout:** `_site/index.html`, `404.html`, `favicon.png`, `app/…`.
  The upload action drops dot-files, so `.streamlit/config.toml` is published
  as `app/streamlit/config.toml` and mounted back at its real path.
- **Caching:** every app file URL carries a content hash (`?v=…`), and the
  manifest is fetched with `cache: "no-cache"`. GitHub Pages caches
  `index.html` for up to 10 minutes.
- **Paths:** all URLs are relative, so the site works under
  `/lake-simulation-timeseries/` and at the root of a custom domain.
- **Routing:** the app is a single page with no client-side routes (the
  documentation view is session state), so refreshing never produces a 404.
  `404.html` links back to the app for mistyped addresses.
- **Runtime pin:** `STLITE_VERSION` in `web/index.html` (1.9.1) fixes the
  Streamlit, Pyodide and Pyodide package versions. Pure-Python packages are
  pinned in `web/requirements.txt`. When upgrading stlite, preview locally
  (`python3 scripts/build_site.py --serve`) and run the demo workbook once.

## 9. Custom-domain configuration

With a GitHub Actions deployment, a `CNAME` file is ignored, so the domain is
configured only in the repository settings. No DNS change is needed to test
the site on the `github.io` URL first.

Current record: `time-series.roland-digital.de. A <Hetzner server IP>` (zone at Hetzner).

When you are ready to switch:

1. *(Recommended, prevents domain takeover)* GitHub → your **Settings → Pages →
   Add a domain** → `roland-digital.de`, then add the `TXT` record GitHub shows
   (`_github-pages-challenge-omidabduli.roland-digital.de`).
2. In the Hetzner DNS zone of `roland-digital.de`:
   - delete the `A` record for `time-series`;
   - add `time-series  CNAME  omidabduli.github.io.` (no repository name; keep
     the trailing dot if the Hetzner console expects fully qualified names).
   - No `AAAA` record exists; do not add one for this subdomain.
3. Repository **Settings → Pages → Custom domain** → `time-series.roland-digital.de` → Save.
4. Wait for the DNS check to pass and the certificate to be issued (minutes, up
   to 24 h), then tick **Enforce HTTPS**.
5. Update the repository description's website URL if needed (it already
   points to `https://time-series.roland-digital.de`).

Afterwards, `https://omidabduli.github.io/lake-simulation-timeseries/`
redirects to the custom domain. Do not use a wildcard DNS record.

## 10. Manual steps still to do

1. **Rotate the Hetzner/CloudPanel password** (see section 7).
2. Review and commit the prepared changes, then push:
   ```bash
   git add -A
   git commit -m "Migrate to GitHub Pages (in-browser Streamlit via stlite)"
   git push origin main
   ```
3. **Enable GitHub Pages:** repository **Settings → Pages → Build and deployment → Source: GitHub Actions**.
   If the first workflow run started before this and failed at "Configure GitHub
   Pages", re-run it from the **Actions** tab.
4. Open <https://omidabduli.github.io/lake-simulation-timeseries/> and try the
   demo workbook ("Download Example File" in the sidebar, then upload it).
5. Switch the custom domain (section 9) once you are happy with the result.
6. Shut down the Hetzner deployment: stop the Streamlit process and delete the
   CloudPanel site. Cancel the server if nothing else runs on it.
7. Optional: empty the Trash item `deploy.command (original with hardcoded credentials)`,
   and purge the old commit from history (section 7).

## 11. Rollback to the Hetzner version

The server code path is intact: `app.py` still runs under `streamlit run`
(verified), and the deployment tooling is in `legacy/server/`.

1. Re-create the CloudPanel site for the domain if it no longer exists.
2. Keep your `.env` in the repository root (see `legacy/server/.env.example`)
   and run `legacy/server/deploy.command`.
3. In DNS, replace the `CNAME` for `time-series` with the previous `A` record
   pointing at the server.
4. In **Settings → Pages**, remove the custom domain (or unpublish the site).

Alternatively, run the container anywhere with
`docker compose -f legacy/server/docker-compose.yml up --build -d`.

## Known limitations and differences

- **First visit** downloads the in-browser Python runtime (about 50 MB,
  mostly SciPy and Plotly). Later visits load from the browser cache, and
  start-up then takes about 10 s. A start-up screen explains this.
- A modern browser with WebAssembly is required (current Chrome, Edge,
  Firefox, Safari). Computation runs on the visitor's device, so speed
  depends on their hardware.
- The browser console shows one harmless `Download Button fetch error` per
  download button. Streamlit 1.62 pre-fetches the file from stlite's internal
  `stlite.invalid` address. Downloads themselves work.
- Unchanged from the server version (verified on both): clicking a download
  button makes Streamlit rerun the script, which returns to the data-quality
  review ("Choose either …") and hides the results until "Proceed as
  provided" is clicked again. Passing `on_click="ignore"` to the download
  buttons would keep the results visible. It was not changed, to preserve behaviour.
- Also pre-existing and identical on the server: the "Download CSV" label is
  dark text on the blue button (low contrast), because the app's global `p`
  colour rule overrides the button's white text colour.
