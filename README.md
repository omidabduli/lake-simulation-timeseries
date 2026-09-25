# Universal Time-Series Forecasting

[![Deploy to GitHub Pages](https://github.com/omidabduli/lake-simulation-timeseries/actions/workflows/deploy-pages.yml/badge.svg)](https://github.com/omidabduli/lake-simulation-timeseries/actions/workflows/deploy-pages.yml)

**Live: https://omidabduli.github.io/lake-simulation-timeseries/**

You give it an Excel file with the past and one or more "what if" scenarios for the future. It learns how your variables move together, predicts the one you left empty, and shows you which inputs drove each prediction. It runs completely in your browser, so your data never leaves your computer.

## Why I built this

I love prediction. Building something that tells you what the future might look like makes almost any subject interesting to me, whether it is a lake, a market or a machine.

But a prediction is only worth something if it is honest. To predict well you have to be realistic: look at how a system actually works, not how you would like it to work. And even then you will often be wrong. What I like about forecasting is that it doesn't let you hide from that. You write down what you expect, reality answers, and you learn something. Every time, the next prediction gets a little closer to how things really are.

That is why I didn't want a black box. A model that says "oxygen will drop" without saying why is a number you can't learn from. So every forecast here comes with an explanation of which inputs drove it, and you can check whether the model learned something real or just something convenient.

I also wanted it to be open to anyone with a spreadsheet. You shouldn't need to write Python, and you shouldn't need a data science team to ask "what happens to my lake if the summer is two degrees warmer?"

## What it does

1. **Upload a workbook.** The first sheet is your history. Every other sheet is a scenario with the same columns, and one column left completely empty. That empty column is what the app predicts.
2. **Check the data.** It looks for placeholder values like `-999` and `-9999` and for outliers (using the interquartile range), shows you what it found, and lets you decide per column whether to clean it.
3. **Train.** It tries a set of XGBoost models and scores each one only on data that comes *after* the data it was trained on. You can watch the search live.
4. **Read the result.** You see the forecast next to the history, the validation score (R² and RMSE), and a SHAP chart of which inputs mattered most.
5. **Export.** Download the predictions as CSV (`;` as separator, `,` as decimal mark, so it opens directly in a German Excel).

There is a demo workbook in `Example/Lake_Demo_2000_Rows.xlsx`: 2,000 days of lake measurements (air and water temperature, rain, wind, sunlight, lake level, phosphorus, chlorophyll and dissolved oxygen) and a 365-day warming scenario in which dissolved oxygen is left empty. Upload it and everything runs by itself.

## What I found

On the demo workbook the best model reaches a cross-validated R² of 0.98 with an RMSE of 0.19 mg/L for dissolved oxygen. SHAP shows that water temperature is by far the biggest driver, followed by chlorophyll. That matches the physics: warm water holds less oxygen, and algae change how much oxygen there is.

I want to be clear about what that number means. The demo data is synthetic and much cleaner than real measurements, so the high score shows that the pipeline works, not that lakes are easy to predict. On real data, expect lower scores, and trust the validation score, never the training fit.

A few things this tool is honestly not good at:

- **It forecasts scenarios, not the unknown.** You have to supply the future values of every other variable. The answer is only as realistic as the scenario you give it.
- **Tree models don't extrapolate.** If a scenario goes beyond anything in the history (for example water warmer than ever measured), XGBoost will treat it like the most extreme case it has seen. The prediction flattens instead of following the trend.
- **One number, no range.** It gives a single forecast per time step, without a confidence interval.

## How it works

**Features.** Months, hours and days of the year are placed on a circle with sine and cosine, so December and January end up next to each other instead of eleven steps apart. For every input the app adds rolling means over the last 3 and 7 observations (14 and 30 in advanced mode, for systems that react slowly). A continuous time index keeps the long-term trend going into the scenarios instead of restarting at zero. The target itself is never used as an input.

**Model search.** A random search over learning rate (0.01 to 0.2), tree depth and number of trees (100 to 500). With the default settings it tests 15 candidates on 4 forward-only folds (scikit-learn's `TimeSeriesSplit`). Normal cross-validation shuffles the rows, which would let the model train on the future to predict the past and make it look better than it is. The search uses a fixed seed, so the same file always gives the same model.

**Explanations.** TreeSHAP measures how much each input moved each prediction, and the chart ranks the inputs by their average absolute impact. When the `shap` package isn't available (it can't be installed in the browser), the app uses XGBoost's own TreeSHAP, which gives the same values.

**No server.** The live site uses [stlite](https://github.com/whitphx/stlite), which runs the unchanged Streamlit app on Pyodide (Python compiled to WebAssembly) inside the browser. Reading the file, training, SHAP and the charts all happen on your device. The first visit downloads about 50 MB of Python runtime; after that it loads from the browser cache.

Each run also writes a small JSON log with the score, the settings and the top SHAP drivers (to `logs/` when running locally).

## Run it yourself

With Python 3.9 or newer:

```bash
pip install -r requirements.txt
streamlit run app.py
```

On macOS, XGBoost needs OpenMP (`brew install libomp`). You can also double-click `run.command`, which starts the app on port 8502.

With Docker:

```bash
docker compose -f docker/docker-compose.yml up --build -d
```

Then open http://localhost:8501.

To build and preview the static browser version:

```bash
python3 scripts/build_site.py --serve
```

Then open http://127.0.0.1:8000/. The tests run with:

```bash
python -m unittest discover -s tests -v
```

Every push to `main` runs the tests, builds the site with `scripts/build_site.py` and publishes it to GitHub Pages (`.github/workflows/deploy-pages.yml`). A failing test stops the deployment.

## Your own data

- **Sheet 1 (history):** one time column named `Time`, `Date`, `Datetime`, `Timestamp`, `Datum` or `Zeit`. All other columns numeric.
- **Sheet 2 and later (scenarios):** exactly the same columns as sheet 1, with one column left completely empty. That is the one it predicts.

There is also a longer explanation in German (`docs/documentation_de.md`) and short, simple introductions in German and Persian (`docs/explanation_de.md`, `docs/explanation_fa.md`).

## License

Apache 2.0, see [LICENSE](LICENSE). Built with Streamlit, XGBoost, SHAP, scikit-learn, pandas, Plotly and stlite.

Made by [Omid Abduli](https://github.com/omidabduli).
