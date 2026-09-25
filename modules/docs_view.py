"""
modules/docs_view.py — Technical documentation page
Universal Time-Series Forecasting
"""
from __future__ import annotations

import streamlit as st

def render_documentation(example_file_bytes: bytes | None = None) -> None:
    """Render the documentation section inside Streamlit."""
    
    # ── Header Lockup (Matching main app landing aesthetic) ────────────────────
    st.markdown(
        """
        <style>
          .doc-hero-shell {
            position: relative;
            width: 100%;
            padding: 2.2rem 2.5rem;
            border: 1px solid var(--border-light, #E6E2D8);
            border-radius: 16px;
            background:
              linear-gradient(rgba(230, 226, 216, 0.35) 1px, transparent 1px),
              linear-gradient(90deg, rgba(230, 226, 216, 0.35) 1px, transparent 1px),
              var(--bg-surface, #F2EFE7);
            background-size: 28px 28px;
            background-position: -1px -1px;
            overflow: hidden;
            margin-bottom: 1.8rem;
            box-shadow: var(--shadow-card, 0 2px 8px rgba(0,0,0,0.03));
          }

          .doc-hero-kicker {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            margin-bottom: 0.9rem;
            color: var(--text-primary, #191919);
            font-family: 'Space Grotesk', 'JetBrains Mono', monospace;
            font-size: 11px;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            font-weight: 600;
          }
          .doc-hero-kicker::before {
            content: "";
            width: 24px;
            height: 1px;
            background: var(--text-primary, #191919);
          }

          .doc-hero-shell h1 {
            max-width: 760px;
            margin: 0;
            color: var(--text-primary, #191919);
            font-family: 'Newsreader', Georgia, serif !important;
            font-size: clamp(32px, 4.5vw, 52px) !important;
            font-weight: 600 !important;
            line-height: 1.1 !important;
            letter-spacing: -0.02em !important;
          }

          .doc-hero-desc {
            max-width: 680px;
            margin: 1.1rem 0 0;
            color: var(--text-secondary, #555555);
            font-family: 'Inter', sans-serif;
            font-size: 15px;
            line-height: 1.55;
          }

          .doc-meta-pills {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-top: 1.4rem;
          }
          .doc-meta-pill {
            padding: 0.35rem 0.65rem;
            border-radius: 9999px;
            border: 1px solid var(--border-light, #E6E2D8);
            background: var(--bg-badge, #191919);
            font-family: 'Inter', sans-serif;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.02em;
            color: var(--on-badge, #FFFFFF);
          }
        </style>

        <div class="doc-hero-shell">
            <div class="doc-hero-kicker">UNIVERSAL TIME-SERIES FORECASTING · TECHNICAL REFERENCE</div>
            <h1>Documentation</h1>
            <p class="doc-hero-desc">
                Workbook format, model and validation method, feature engineering, SHAP explanations, metrics, code layout and deployment.
            </p>
            <div class="doc-meta-pills">
                <span class="doc-meta-pill">v2.0.0</span>
                <span class="doc-meta-pill">XGBoost Regressor</span>
                <span class="doc-meta-pill">TreeSHAP Interpretability</span>
                <span class="doc-meta-pill">TimeSeriesSplit CV</span>
                <span class="doc-meta-pill">IQR Auditing</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Sub-tab navigation ────────────────────────────────────────────────────
    tab_names = [
        "01. Overview",
        "02. Workbook Schema",
        "03. ML Architecture",
        "04. Feature Engineering",
        "05. Explainable AI",
        "06. Evaluation Metrics",
        "07. Codebase Architecture",
        "08. Deployment",
        "09. Troubleshooting",
    ]
    
    (
        tab_overview,
        tab_workbook,
        tab_ml,
        tab_fe,
        tab_shap,
        tab_metrics,
        tab_python,
        tab_docker,
        tab_faq,
    ) = st.tabs(tab_names)

    # =========================================================================
    # TAB 1: OVERVIEW & QUICK START
    # =========================================================================
    with tab_overview:
        st.markdown(
            r"""
            ### Overview
            Universal Time-Series Forecasting predicts one variable of a multi-variable time series for future scenarios. You provide historical observations and one or more scenario sheets in which the target column is empty. The app trains an XGBoost model on the history, fills in the target for each scenario and uses SHAP to show which inputs drove the forecast. The lake dataset is only a demo; any regularly sampled time series in the same layout works.

            #### Main parts
            """,
            unsafe_allow_html=True,
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                """
                <div class="eco-card">
                    <h4 style="margin: 0 0 0.5rem 0; color: var(--on-surface); font-size: 15px;">Data preparation</h4>
                    <p style="margin: 0; font-size: 13px; color: var(--on-surface-variant); line-height: 1.5;">
                        Reads multi-sheet Excel workbooks, finds the time column, detects the prediction target ($y$) and flags placeholder values such as -999 and statistical outliers.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                """
                <div class="eco-card">
                    <h4 style="margin: 0 0 0.5rem 0; color: var(--on-surface); font-size: 15px;">Time-ordered validation</h4>
                    <p style="margin: 0; font-size: 13px; color: var(--on-surface-variant); line-height: 1.5;">
                        Uses expanding-window cross-validation (<code>TimeSeriesSplit</code>), so the model is always validated on data that comes after its training data.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
                """
                <div class="eco-card">
                    <h4 style="margin: 0 0 0.5rem 0; color: var(--on-surface); font-size: 15px;">SHAP explanations</h4>
                    <p style="margin: 0; font-size: 13px; color: var(--on-surface-variant); line-height: 1.5;">
                        Computes TreeSHAP values for the scenario predictions and ranks the inputs by their average contribution (in the lake demo, for example air temperature or solar radiation).
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown("### Workflow")
        
        st.markdown(
            """
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 0.85rem; margin: 1rem 0 1.5rem 0;">
                <div style="background: var(--surface-container); border: 1px solid var(--border); padding: 1.1rem; border-radius: var(--rounded-md);">
                    <div style="font-family: 'Space Grotesk', monospace; color: var(--primary); font-weight: 700; font-size: 12px;">STEP 01</div>
                    <h4 style="margin: 0.3rem 0; font-size: 14px; color: var(--on-surface);">Upload Workbook</h4>
                    <p style="margin: 0; font-size: 12px; color: var(--on-surface-variant);">Upload an Excel file with one historical sheet followed by one or more scenario sheets.</p>
                </div>
                <div style="background: var(--surface-container); border: 1px solid var(--border); padding: 1.1rem; border-radius: var(--rounded-md);">
                    <div style="font-family: 'Space Grotesk', monospace; color: var(--primary); font-weight: 700; font-size: 12px;">STEP 02</div>
                    <h4 style="margin: 0.3rem 0; font-size: 14px; color: var(--on-surface);">Audit Data Quality</h4>
                    <p style="margin: 0; font-size: 12px; color: var(--on-surface-variant);">Review placeholder values (e.g. <code>-999</code>) and IQR outlier flags before training.</p>
                </div>
                <div style="background: var(--surface-container); border: 1px solid var(--border); padding: 1.1rem; border-radius: var(--rounded-md);">
                    <div style="font-family: 'Space Grotesk', monospace; color: var(--primary); font-weight: 700; font-size: 12px;">STEP 03</div>
                    <h4 style="margin: 0.3rem 0; font-size: 14px; color: var(--on-surface);">Train</h4>
                    <p style="margin: 0; font-size: 12px; color: var(--on-surface-variant);">A random search tests XGBoost settings with time-ordered cross-validation and keeps the best one.</p>
                </div>
                <div style="background: var(--surface-container); border: 1px solid var(--border); padding: 1.1rem; border-radius: var(--rounded-md);">
                    <div style="font-family: 'Space Grotesk', monospace; color: var(--primary); font-weight: 700; font-size: 12px;">STEP 04</div>
                    <h4 style="margin: 0.3rem 0; font-size: 14px; color: var(--on-surface);">Review results</h4>
                    <p style="margin: 0; font-size: 12px; color: var(--on-surface-variant);">Inspect the forecast chart, the validation scores and the SHAP feature ranking.</p>
                </div>
                <div style="background: var(--surface-container); border: 1px solid var(--border); padding: 1.1rem; border-radius: var(--rounded-md);">
                    <div style="font-family: 'Space Grotesk', monospace; color: var(--primary); font-weight: 700; font-size: 12px;">STEP 05</div>
                    <h4 style="margin: 0.3rem 0; font-size: 14px; color: var(--on-surface);">Export</h4>
                    <p style="margin: 0; font-size: 12px; color: var(--on-surface-variant);">Download the results as CSV. Each run is also written to a JSON log file.</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if example_file_bytes:
            st.markdown("#### Demo dataset")
            st.markdown(
                "The demo workbook contains 2,000 daily lake observations and a 365-day warming scenario. Use it as a template for your own data."
            )
            st.download_button(
                label="Download demo workbook (.xlsx)",
                data=example_file_bytes,
                file_name="Lake_Demo_2000_Rows.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="doc_download_demo",
            )

    # =========================================================================
    # TAB 2: WORKBOOK & DATA SPECIFICATION
    # =========================================================================
    with tab_workbook:
        st.markdown(
            r"""
            ### Workbook format
            The app reads an Excel workbook (`.xlsx` or `.xls`) with one table per sheet: a time column plus one column per variable.

            #### 1. Sheet order
            * **Sheet 1 (historical)**: past observations at a regular interval (for example daily or hourly), including the target variable.
            * **Sheets 2 and later (scenarios)**: the same column names as sheet 1. Each sheet describes one possible future, for example a warming scenario or a wet year.

            **Target column**: in every scenario sheet, exactly one column must be left completely empty. The app uses this column as the target ($y$). If no column or more than one column is empty, the scenario is rejected with an error message.

            ---
            #### 2. Time column
            The time column is found by its header, ignoring case: `Time`, `Date`, `Datetime`, `Timestamp`, `Datum` or `Zeit`. If none of these is present, the first column is used when it contains dates, otherwise any column with a date type. The rows are then sorted by time.

            Date formats:
            * Cells formatted as dates in Excel are read directly. This is the most reliable option.
            * Text dates are parsed with `pandas.to_datetime`, for example `2024-06-15` or `2024-06-15 14:30:00`.
            * Day-first text such as `01.02.2024` can be read as month-first, and mixed formats in one column are not parsed. Use ISO format (`YYYY-MM-DD`) or real Excel dates.

            ---
            #### 3. Placeholder values
            Many loggers write a fixed number such as -999 when a measurement is missing. These values are listed in the data quality panel. For each column you can choose to replace them with empty values:
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            | Value | Source | Treatment |
            | :--- | :--- | :--- |
            | `-999`, `-9999` | Default list | Replaced with `NaN` if selected, then forward-filled |
            | `999`, `9999` | Default list | Replaced with `NaN` if selected, then forward-filled |
            | Custom values | Entered by the user (e.g. `-1`) | Added in the advanced settings, then treated the same way |

            Empty values are filled forward during feature engineering. Gaps at the very start of a series are filled backward.
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div style="padding: 0.85rem 1.1rem; background: var(--surface-container); border-left: 3px solid var(--primary); border-radius: var(--rounded-md); margin-top: 1rem;">
                <strong style="color: var(--on-surface); font-size: 13px;">Negative values</strong>
                <p style="margin: 0.2rem 0 0 0; font-size: 13px; color: var(--on-surface-variant);">
                    Many measurements can be negative, for example air temperature (-5.2 °C). For that reason -1 is not treated as missing by default. Add it only if your data uses -1 as a placeholder.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # =========================================================================
    # TAB 3: MACHINE LEARNING ARCHITECTURE & CV
    # =========================================================================
    with tab_ml:
        st.markdown(
            r"""
            ### Model and validation

            #### XGBoost regressor
            The forecasting model is an XGBoost regressor (Extreme Gradient Boosting), an ensemble of decision trees that works well on tabular data.

            Gradient boosting reduces the loss step by step. Each new tree is fitted to the remaining errors of the trees before it:
            $$F_m(x) = F_{m-1}(x) + \gamma_m h_m(x)$$
            where $h_m(x)$ is the tree added at step $m$ and $\gamma_m$ is its weight, scaled by the learning rate.

            #### Hyperparameter search
            The app runs a random search. Each candidate draws a learning rate between 0.01 and 0.2, a number of trees between 100 and 500, and a maximum tree depth from the range set by the model complexity option (2 to 5, 2 to 8 or 2 to 10). Search effort sets the number of candidates (8, 15 or 30). Every candidate is scored with time-series cross-validation, and the one with the highest mean validation $R^2$ is retrained on the full history.

            ---
            #### Time-series cross-validation (`TimeSeriesSplit`)
            Standard K-fold cross-validation shuffles the rows, so the model can be trained on later data and tested on earlier data. For time series this leaks information and makes the scores look better than they are.

            The app uses expanding-window cross-validation instead. Each fold trains on all data up to a cut-off and validates on the block that follows. The number of folds can be set to 3, 4 or 5 (default 4). Schematic example:
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            r"""
            ```text
            Fold 1: [ Train: Step 1..400 ]  --> [ Validate: Step 401..500 ]
            Fold 2: [ Train: Step 1..500 ]  --> [ Validate: Step 501..600 ]
            Fold 3: [ Train: Step 1..600 ]  --> [ Validate: Step 601..700 ]
            Fold 4: [ Train: Step 1..700 ]  --> [ Validate: Step 701..800 ]
            ```
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            r"""
            ---
            #### Outlier check (interquartile range)
            The data quality check flags values in every numeric column using the interquartile range (IQR) rule:
            $$IQR = Q_3 - Q_1$$
            $$\text{Lower Fence} = Q_1 - k \cdot IQR, \quad \text{Upper Fence} = Q_3 + k \cdot IQR$$

            * **$k = 3.0$ (default)**: flags only far-out values, so real but unusual events are mostly kept.
            * **$k = 1.5$**: stricter, useful for quiet, low-noise series.

            $k$ can be set between 1.5 and 5.0. Flagged values are only listed. They are replaced with empty values only for the columns you select.
            """,
            unsafe_allow_html=True,
        )

    # =========================================================================
    # TAB 4: FEATURE ENGINEERING & SEASONALITY
    # =========================================================================
    with tab_fe:
        st.markdown(
            r"""
            ### Feature engineering

            #### Cyclical time features
            Time stored as integers (month 1 to 12, hour 0 to 23) has a jump at the end of each cycle: December (12) and January (1) look far apart, although they are neighbours.

            The app therefore places calendar time on a unit circle with sine and cosine. These features can be switched off with the seasonality option.

            ##### Month ($T = 12$)
            $$\text{month\_sin} = \sin\left(\frac{2\pi \cdot \text{month}}{12}\right), \quad \text{month\_cos} = \cos\left(\frac{2\pi \cdot \text{month}}{12}\right)$$

            ##### Hour of day ($T = 24$)
            $$\text{hour\_sin} = \sin\left(\frac{2\pi \cdot \text{hour}}{24}\right), \quad \text{hour\_cos} = \cos\left(\frac{2\pi \cdot \text{hour}}{24}\right)$$

            ##### Day of year ($T = 365$)
            $$\text{doy\_sin} = \sin\left(\frac{2\pi \cdot \text{doy}}{365}\right), \quad \text{doy\_cos} = \cos\left(\frac{2\pi \cdot \text{doy}}{365}\right)$$

            #### Trend features
            Two features let the model follow long-term trends: `year` and `time_idx_days`, the number of days since the first historical record. Scenario sheets use the same starting point, so the day count continues instead of restarting at zero.

            ---
            #### Rolling means
            Many systems respond to their inputs with a delay. To give the model this context, the app adds a rolling mean of every numeric input column (the target is excluded):
            $$\mu_{w}(t) = \frac{1}{w}\sum_{i=0}^{w-1} x_{t-i}$$
            The windows can be chosen from 3, 7, 14 and 30 observations (default 3 and 7). At the start of a series the mean uses the observations available so far.

            Rolling means are computed within each sheet. The first rows of a scenario therefore do not use values from the end of the historical sheet.
            """,
            unsafe_allow_html=True,
        )

    # =========================================================================
    # TAB 5: EXPLAINABLE AI (SHAP ENGINE)
    # =========================================================================
    with tab_shap:
        st.markdown(
            r"""
            ### SHAP explanations

            #### Why explanations matter
            A forecast is easier to trust and to check when you can see which inputs drove it.

            The app uses TreeSHAP (SHapley Additive exPlanations). It is based on Shapley values from cooperative game theory and computes the contribution of each input to each prediction exactly for tree models.

            ---
            #### Shapley values
            For a feature set $F$ and a specific feature $i$, the Shapley value $\phi_i(x)$ is calculated across all feature subsets $S \subseteq F \setminus \{i\}$:
            $$\phi_i(x) = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F| - |S| - 1)!}{|F|!} \Big( f_x(S \cup \{i\}) - f_x(S) \Big)$$

            ##### Properties
            1. **Local accuracy**: The sum of all feature SHAP values equals the difference between the model forecast $f(x)$ and the baseline expected value $E[f(x)]$:
               $$f(x) = E[f(x)] + \sum_{i=1}^M \phi_i(x)$$
            2. **Consistency**: if a feature's marginal contribution increases or stays the same, its SHAP value does not decrease.
            3. **Symmetry**: features that contribute equally to all subsets receive equal SHAP values.

            ---
            #### Reading the SHAP chart
            * **Bar chart**: the app computes SHAP values for all scenario predictions and ranks the inputs by their mean absolute value ($\frac{1}{N}\sum |\phi_i|$). Longer bars mean a larger average influence on the forecast.
            * **Sign**: for a single prediction, a positive SHAP value pushes the forecast up and a negative value pushes it down. The bar chart uses absolute values, so it shows size, not direction.
            """,
            unsafe_allow_html=True,
        )

    # =========================================================================
    # TAB 6: FORECASTING & EVALUATION METRICS
    # =========================================================================
    with tab_metrics:
        st.markdown(
            r"""
            ### Evaluation metrics

            The app reports two metrics. Both are averaged over the cross-validation folds of the selected model, so they describe performance on data the model was not trained on.

            #### 1. Coefficient of determination ($R^2$)
            The share of the variance in the target that the model explains:
            $$R^2 = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n (y_i - \bar{y})^2}$$
            * **1.0**: perfect prediction.
            * **0.0**: no better than always predicting the mean $\bar{y}$.
            * **Below 0.0**: worse than predicting the mean.

            ---
            #### 2. Root mean squared error (RMSE)
            The typical size of the prediction error, in the units of the target. Large errors count more than small ones because they are squared:
            $$RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2}$$
            Lower is better.
            """,
            unsafe_allow_html=True,
        )

    # =========================================================================
    # TAB 7: PYTHON CODEBASE ARCHITECTURE
    # =========================================================================
    with tab_python:
        st.markdown(
            r"""
            ### Code structure

            `app.py` holds the Streamlit interface and runs the forecasting steps. The individual steps live in separate modules.

            ```text
            lake-simulation-timeseries/
            ├── app.py                      # Streamlit interface, search and cross-validation
            ├── web/                        # GitHub Pages host page and browser entry point (stlite)
            ├── scripts/build_site.py       # Builds the static site for GitHub Pages
            ├── docker/                     # Dockerfile and docker-compose.yml
            ├── modules/
            │   ├── data_loader.py          # Excel reading, sheet checks, data quality
            │   ├── feature_engineering.py  # Seasonal, trend and rolling features
            │   ├── model.py                # XGBoost training and prediction
            │   ├── explainer.py            # TreeSHAP values
            │   ├── visualizer.py           # Plotly charts
            │   ├── logger.py               # JSON run logs
            │   └── docs_view.py            # This documentation page
            ```

            ---
            #### Modules
            * `modules/data_loader.py`: reads the workbook (`load_excel`), splits it into history and scenarios (`separate_sheets`), sets the time index (`normalize_time_index`), detects the target column (`detect_target_column`), checks that columns match (`validate_column_match`), flags placeholders and IQR outliers (`detect_data_issues`) and replaces selected values with `NaN` (`clean_specific_columns`).
            * `modules/feature_engineering.py`: adds the sine/cosine features for month, hour and day of year, the trend features and the rolling means (`engineer_features`).
            * `modules/model.py`: default XGBoost settings and the `train` and `predict` functions. The random search over `TimeSeriesSplit` folds runs in `app.py`.
            * `modules/explainer.py`: computes mean absolute SHAP values for the scenario predictions with `shap.TreeExplainer`. In the browser build, where `shap` cannot be installed, XGBoost's built-in TreeSHAP gives the same values.
            * `modules/visualizer.py`: Plotly charts for the historical data and scenario forecast, and the SHAP bar chart.
            * `modules/logger.py`: writes one JSON file per run to `logs/<scenario>_YYYYMMDD_HHMMSS.json` with the target, the metrics and the top SHAP features.
            """,
            unsafe_allow_html=True,
        )

    # =========================================================================
    # TAB 8: DOCKER & PRODUCTION DEPLOYMENT
    # =========================================================================
    with tab_docker:
        st.markdown(
            r"""
            ### Deployment

            #### Option A: GitHub Pages
            The public version is a static site. The Python code runs in the visitor's browser through [stlite](https://github.com/whitphx/stlite) (Streamlit on Pyodide/WebAssembly). There is no server, and uploaded workbooks stay on the visitor's device.

            Every push to `main` runs `.github/workflows/deploy-pages.yml`, which runs the tests, builds the site with `scripts/build_site.py` and publishes it to GitHub Pages. To preview the build locally:

            ```bash
            python3 scripts/build_site.py --serve
            # Open http://127.0.0.1:8000/
            ```

            ---
            #### Option B: Docker
            Runs the app as a Streamlit server in a container. The Docker files are in `docker/`. Run the commands from the repository root:

            ```bash
            # 1. Clone the repository
            git clone https://github.com/omidabduli/lake-simulation-timeseries.git
            cd lake-simulation-timeseries

            # 2. Build and start the container in the background
            docker compose -f docker/docker-compose.yml up --build -d

            # 3. Open http://localhost:8501
            ```

            Without Compose:
            ```bash
            docker build -f docker/Dockerfile -t time-series-forecasting .
            docker run -d -p 8501:8501 -v "$(pwd)/logs:/app/logs" --name time-series-forecasting time-series-forecasting
            ```

            ---
            #### Option C: Local launcher (macOS / Linux)
            Double-click `run.command` or start it from a terminal. The app opens at http://localhost:8502.
            ```bash
            chmod +x run.command
            ./run.command
            ```
            """,
            unsafe_allow_html=True,
        )

    # =========================================================================
    # TAB 9: FAQ & TROUBLESHOOTING
    # =========================================================================
    with tab_faq:
        st.markdown("### Troubleshooting")

        with st.expander("Why do I get a 'File Structure Error' after uploading?", expanded=True):
            st.markdown(
                """
                **Cause**: the workbook has fewer than two sheets or is missing column headers.  
                **Fix**: put the historical data in the first sheet and the scenarios in the following sheets, with the same column headers in every sheet. The sheet names do not matter.
                """
            )

        with st.expander("How does the app know which variable to predict?", expanded=False):
            st.markdown(
                """
                In each scenario sheet, leave the column you want to forecast empty in every row. The app looks for the one empty column and uses it as the target. If no column or more than one column is empty, the scenario is rejected.
                """
            )

        with st.expander("How are placeholder values like -999 handled?", expanded=False):
            st.markdown(
                """
                Placeholder values listed in the data quality panel are replaced with `NaN` for the columns you select (selected by default). During feature engineering, empty values are filled with the last valid value, and gaps at the start of a series with the next valid value.
                """
            )

        with st.expander("Why is TimeSeriesSplit used instead of standard K-Fold CV?", expanded=False):
            st.markdown(
                """
                K-fold shuffles the rows, so the model can be trained on later dates and tested on earlier ones. This lookahead makes the scores too optimistic. `TimeSeriesSplit` always trains on the past and validates on the period that follows.
                """
            )

        with st.expander("Can I open the results in Excel?", expanded=False):
            st.markdown(
                """
                Yes. The CSV export uses `;` as the delimiter and `,` as the decimal separator, which matches German and most European Excel settings.
                """
            )
