"""
modules/docs_view.py — Technical Documentation & Reference Manual
EcoPredict AI · Lake Time-Series Forecasting & Scenario Simulation Engine
"""
from __future__ import annotations

import streamlit as st

def render_documentation(example_file_bytes: bytes | None = None) -> None:
    """Render the clean, professional documentation section inside Streamlit."""
    
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
            <h1>System Documentation & Reference Manual</h1>
            <p class="doc-hero-desc">
                Complete technical specifications, mathematical foundations, data workbook standards, and architecture reference for the universal forecasting and scenario simulation engine.
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

    # ── Interactive Sub-Tabs Navigation (No Emojis) ───────────────────────────
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
    # TAB 1: EXECUTIVE OVERVIEW & QUICK START
    # =========================================================================
    with tab_overview:
        st.markdown(
            r"""
            ### Executive System Overview
            **Universal Time-Series Forecasting** is an explainable machine learning platform designed for data scientists, financial analysts, operations leaders, engineers, and researchers. It solves the key bottleneck in multi-variable scenario analysis: **converting complex multi-variable historical observations into explainable future predictions across any domain**.

            #### Key Capabilities & Architecture Pillars
            """,
            unsafe_allow_html=True,
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                """
                <div class="eco-card">
                    <h4 style="margin: 0 0 0.5rem 0; color: var(--on-surface); font-size: 15px;">Automated Pipeline</h4>
                    <p style="margin: 0; font-size: 13px; color: var(--on-surface-variant); line-height: 1.5;">
                        Ingests raw multi-sheet Excel workbooks, normalizes heterogeneous timestamp formats, automatically identifies the prediction target ($y$), and handles missing sentinel values.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                """
                <div class="eco-card">
                    <h4 style="margin: 0 0 0.5rem 0; color: var(--on-surface); font-size: 15px;">No-Leakage Cross Validation</h4>
                    <p style="margin: 0; font-size: 13px; color: var(--on-surface-variant); line-height: 1.5;">
                        Uses forward-only expanding window cross-validation (<code>TimeSeriesSplit</code>) to strictly prevent future temporal observations from leaking into past training iterations.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
                """
                <div class="eco-card">
                    <h4 style="margin: 0 0 0.5rem 0; color: var(--on-surface); font-size: 15px;">Game-Theoretic XAI</h4>
                    <p style="margin: 0; font-size: 13px; color: var(--on-surface-variant); line-height: 1.5;">
                        Computes exact TreeSHAP values for every prediction step, surfacing the exact physical drivers (e.g. water temperature, solar radiation, runoff) behind forecasts.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown("### Step-by-Step Workflow")
        
        st.markdown(
            """
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 0.85rem; margin: 1rem 0 1.5rem 0;">
                <div style="background: var(--surface-container); border: 1px solid var(--border); padding: 1.1rem; border-radius: var(--rounded-md);">
                    <div style="font-family: 'Space Grotesk', monospace; color: var(--primary); font-weight: 700; font-size: 12px;">STEP 01</div>
                    <h4 style="margin: 0.3rem 0; font-size: 14px; color: var(--on-surface);">Upload Workbook</h4>
                    <p style="margin: 0; font-size: 12px; color: var(--on-surface-variant);">Upload an <code>.xlsx</code> file containing 1 Historical sheet and 1+ Scenario sheets.</p>
                </div>
                <div style="background: var(--surface-container); border: 1px solid var(--border); padding: 1.1rem; border-radius: var(--rounded-md);">
                    <div style="font-family: 'Space Grotesk', monospace; color: var(--primary); font-weight: 700; font-size: 12px;">STEP 02</div>
                    <h4 style="margin: 0.3rem 0; font-size: 14px; color: var(--on-surface);">Audit Data Quality</h4>
                    <p style="margin: 0; font-size: 12px; color: var(--on-surface-variant);">Review sentinel values (e.g. <code>-999</code>) and IQR outlier flags before training.</p>
                </div>
                <div style="background: var(--surface-container); border: 1px solid var(--border); padding: 1.1rem; border-radius: var(--rounded-md);">
                    <div style="font-family: 'Space Grotesk', monospace; color: var(--primary); font-weight: 700; font-size: 12px;">STEP 03</div>
                    <h4 style="margin: 0.3rem 0; font-size: 14px; color: var(--on-surface);">Train & Optimize</h4>
                    <p style="margin: 0; font-size: 12px; color: var(--on-surface-variant);">Watch live forward-only cross-validation optimize XGBoost hyperparameters.</p>
                </div>
                <div style="background: var(--surface-container); border: 1px solid var(--border); padding: 1.1rem; border-radius: var(--rounded-md);">
                    <div style="font-family: 'Space Grotesk', monospace; color: var(--primary); font-weight: 700; font-size: 12px;">STEP 04</div>
                    <h4 style="margin: 0.3rem 0; font-size: 14px; color: var(--on-surface);">Analyze SHAP Drivers</h4>
                    <p style="margin: 0; font-size: 12px; color: var(--on-surface-variant);">Explore interactive Plotly forecasts alongside SHAP directional feature importances.</p>
                </div>
                <div style="background: var(--surface-container); border: 1px solid var(--border); padding: 1.1rem; border-radius: var(--rounded-md);">
                    <div style="font-family: 'Space Grotesk', monospace; color: var(--primary); font-weight: 700; font-size: 12px;">STEP 05</div>
                    <h4 style="margin: 0.3rem 0; font-size: 14px; color: var(--on-surface);">Export & Telemetry</h4>
                    <p style="margin: 0; font-size: 12px; color: var(--on-surface-variant);">Download German locale CSV results and review structured JSON telemetry logs.</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if example_file_bytes:
            st.markdown("#### Demonstration Dataset")
            st.markdown(
                "Download the demonstration dataset below to inspect a fully-formatted workbook with 2,000 daily observations and a 365-day warming scenario."
            )
            st.download_button(
                label="Download Demo Dataset (.xlsx)",
                data=example_file_bytes,
                file_name="Lake_Time_Series_Forecasting_Demo_2000_Rows.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="doc_download_demo",
            )

    # =========================================================================
    # TAB 2: WORKBOOK & DATA SPECIFICATION
    # =========================================================================
    with tab_workbook:
        st.markdown(
            r"""
            ### Workbook Architecture & Schema Specifications
            To ensure zero-configuration processing, Excel files must follow a clean tabular layout. The forecasting engine automatically parses sheets according to strict limnological data rules.

            #### 1. Sheet Hierarchy Rules
            * **Sheet 1 (Historical Baseline)**: Must contain historical observations recorded at regular time intervals (e.g. daily, hourly). All columns must be complete (or contain handled sentinels).
            * **Sheets 2+ (Future Scenarios)**: Must have the **exact same column names and order** as Sheet 1. Each scenario sheet represents a projected future state (e.g., *Warming Scenario*, *High Rainfall Scenario*).
            
            **Target Column Auto-Detection**: In every scenario sheet, exactly **one variable column must be left completely blank (empty NaN)**. The engine automatically detects this blank column as the target variable ($y$) to predict.

            ---
            #### 2. Temporal Index Standards
            The engine automatically scans columns for a valid date/time header. Supported header names include:
            `Time`, `Date`, `Datetime`, `Timestamp`, `Datum`, `Zeit`, `Tage`, `Jahr_Tag`.

            Supported timestamp formats parsed automatically:
            * `YYYY-MM-DD` (e.g., `2024-06-15`)
            * `YYYY-MM-DD HH:MM:SS` (e.g., `2024-06-15 14:30:00`)
            * `DD.MM.YYYY` (German format, e.g., `15.06.2024`)
            * Standard Excel serial date integers/floats.

            ---
            #### 3. Data Cleaning & Sentinel Values
            Scientific logging instruments frequently encode sensor failures using sentinel values. The platform allows configuring missing value sentinels:
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            | Sentinel Code | Interpretation | Treatment in Pipeline |
            | :--- | :--- | :--- |
            | `-999.0` | Default Sensor Error Code | Converted to `NaN` & linearly/forward interpolated |
            | `-9999.0` | Out-of-Bounds Outlier | Converted to `NaN` & linearly/forward interpolated |
            | `999.0` / `9999.0` | Telemetry Overflow Flag | Converted to `NaN` & linearly/forward interpolated |
            | Custom values | User-defined (e.g. `-1`) | Added via Missing-Value Rules UI panel |
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div style="padding: 0.85rem 1.1rem; background: var(--surface-container); border-left: 3px solid var(--primary); border-radius: var(--rounded-md); margin-top: 1rem;">
                <strong style="color: var(--on-surface); font-size: 13px;">Note on Negative Values</strong>
                <p style="margin: 0.2rem 0 0 0; font-size: 13px; color: var(--on-surface-variant);">
                    Physical measurements like air temperature or net heat flux can naturally be negative (e.g., -5.2 °C). Therefore, -1 is NOT treated as a missing sentinel by default. Only enable -1 if your instrument explicitly uses it as an error flag.
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
            ### Machine Learning Core & Validation Rigor

            #### XGBoost Regressor Engine
            The core predictive model is powered by **XGBoost (Extreme Gradient Boosting)**, an ensemble decision tree framework renowned for state-of-the-art performance on tabular time-series data. 

            Gradient boosting minimizes the loss function iteratively by fitting new trees to the residual errors of prior iterations:
            $$F_m(x) = F_{m-1}(x) + \gamma_m h_m(x)$$
            where $h_m(x)$ is the base tree learner added at step $m$, and $\gamma_m$ is the tree shrinkage weight (learning rate).

            ---
            #### Time-Series Cross Validation (`TimeSeriesSplit`)
            Standard K-Fold cross-validation randomly shuffles data, causing **catastrophic temporal leakage** (using tomorrow's water temperature to predict yesterday's dissolved oxygen). 

            To prevent leakage, our engine uses **Forward-Only Expanding Window Cross Validation**:
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
            #### Statistical Outlier Auditing (Interquartile Range)
            To prevent severe instrument anomalies from corrupting tree splits, the data quality module scans all numeric predictors using the **Interquartile Range (IQR)** rule:
            $$IQR = Q_3 - Q_1$$
            $$\text{Lower Fence} = Q_1 - k \cdot IQR, \quad \text{Upper Fence} = Q_3 + k \cdot IQR$$

            * **Extreme Outliers ($k = 3.0$, Default)**: Filters severe telemetry or data corruptions while preserving legitimate extreme real-world events (e.g. market volatility, weather spikes, surge demand).
            * **Mild Outliers ($k = 1.5$, Optional)**: Stricter filter suitable for quiet, low-noise time-series signals.
            """,
            unsafe_allow_html=True,
        )

    # =========================================================================
    # TAB 4: FEATURE ENGINEERING & SEASONALITY
    # =========================================================================
    with tab_fe:
        st.markdown(
            r"""
            ### Feature Engineering & Temporal Seasonality

            #### Cyclical Trigonometric Encodings
            Integer representations of time (e.g., Month 1 to 12, Hour 0 to 23) create artificial discontinuities—Month 12 (December) and Month 1 (January) appear numerically far apart despite being adjacent seasons.

            Our engine maps calendar time onto a 2D continuous unit circle using Sine and Cosine transformations:

            ##### Monthly Seasonality ($T = 12$ months)
            $$\text{month\_sin} = \sin\left(\frac{2\pi \cdot \text{month}}{12}\right), \quad \text{month\_cos} = \cos\left(\frac{2\pi \cdot \text{month}}{12}\right)$$

            ##### Hourly Seasonality ($T = 24$ hours)
            $$\text{hour\_sin} = \sin\left(\frac{2\pi \cdot \text{hour}}{24}\right), \quad \text{hour\_cos} = \cos\left(\frac{2\pi \cdot \text{hour}}{24}\right)$$

            ##### Annual Day-of-Year Seasonality ($T = 365.25$ days)
            $$\text{day\_sin} = \sin\left(\frac{2\pi \cdot \text{day}}{365.25}\right), \quad \text{day\_cos} = \cos\left(\frac{2\pi \cdot \text{day}}{365.25}\right)$$

            ---
            #### Continuous Lag & Rolling Window Aggregates
            Many physical, financial, and operational systems exhibit temporal inertia (e.g. system outputs take time to respond to external drivers). The feature pipeline computes:
            1. **Lag Features**: $x_{t-1}, x_{t-2}, x_{t-3}$ capturing immediate short-term momentum.
            2. **Rolling Window Means**: $\mu_{w}(t) = \frac{1}{w}\sum_{i=0}^{w-1} x_{t-i}$ over 3-step, 7-step, and 14-step windows.
            3. **Rolling Standard Deviations**: $\sigma_{w}(t)$ measuring short-term variance.

            **Scenario Timeline Continuity**: When forecasting future scenarios, rolling windows seamlessly anchor to the tail of the historical dataset, ensuring zero boundary artifact at the start of prediction.
            """,
            unsafe_allow_html=True,
        )

    # =========================================================================
    # TAB 5: EXPLAINABLE AI (SHAP ENGINE)
    # =========================================================================
    with tab_shap:
        st.markdown(
            r"""
            ### Explainable AI (XAI) & TreeSHAP Valuation

            #### Why Explainability Matters in Forecasting
            Machine learning models are frequently criticized as "black boxes." In scenario modeling, stakeholders must understand *why* a model projects a specific outcome.

            Our engine integrates **TreeSHAP (SHapley Additive exPlanations)**, a game-theoretic framework that computes the exact marginal contribution of each variable to every single prediction.

            ---
            #### Game Theory Mathematical Foundation
            For a feature set $F$ and a specific feature $i$, the Shapley value $\phi_i(x)$ is calculated across all feature subsets $S \subseteq F \setminus \{i\}$:
            $$\phi_i(x) = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F| - |S| - 1)!}{|F|!} \Big( f_x(S \cup \{i\}) - f_x(S) \Big)$$

            ##### Key Theoretical Guarantees:
            1. **Efficiency / Local Accuracy**: The sum of all feature SHAP values equals the difference between the model forecast $f(x)$ and the baseline expected value $E[f(x)]$:
               $$f(x) = E[f(x)] + \sum_{i=1}^M \phi_i(x)$$
            2. **Consistency**: If a feature's marginal contribution increases or stays constant, its assigned SHAP value will never decrease.
            3. **Symmetry**: Features contributing equally to all coalitions receive equal SHAP values.

            ---
            #### How to Interpret SHAP Outputs in the App
            * **Mean Absolute SHAP Bar Chart**: Ranks predictors by overall impact ($\frac{1}{N}\sum |\phi_i|$). Longer bars indicate primary drivers of system dynamics.
            * **Directional SHAP Impact**:
              * **Positive SHAP Value (+)**: Pushes the predicted target variable higher.
              * **Negative SHAP Value (-)**: Drives the predicted target variable lower.
            """,
            unsafe_allow_html=True,
        )

    # =========================================================================
    # TAB 6: FORECASTING & EVALUATION METRICS
    # =========================================================================
    with tab_metrics:
        st.markdown(
            r"""
            ### Forecasting Performance & Scientific Evaluation Metrics

            During cross-validation and final model evaluation, four standardized performance metrics are computed to evaluate model fidelity.

            #### 1. Coefficient of Determination ($R^2$)
            Measures the proportion of variance in the target variable explained by the model:
            $$R^2 = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n (y_i - \bar{y})^2}$$
            * **Score = 1.0**: Perfect prediction.
            * **Score = 0.0**: Model performs no better than predicting the mean $\bar{y}$.
            * **Score < 0.0**: Model performs worse than the simple historical average.

            ---
            #### 2. Root Mean Squared Error (RMSE)
            Penalizes large errors heavily, giving an accurate measure of prediction error magnitude in original target units:
            $$RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2}$$

            ---
            #### 3. Mean Absolute Error (MAE)
            Provides the average linear magnitude of errors without over-emphasizing extreme outliers:
            $$MAE = \frac{1}{n}\sum_{i=1}^n |y_i - \hat{y}_i|$$

            ---
            #### 4. Mean Absolute Percentage Error (MAPE)
            Expresses prediction error as a percentage relative to actual physical values:
            $$MAPE = \frac{100\%}{n}\sum_{i=1}^n \left|\frac{y_i - \hat{y}_i}{y_i}\right|$$
            """,
            unsafe_allow_html=True,
        )

    # =========================================================================
    # TAB 7: PYTHON CODEBASE ARCHITECTURE
    # =========================================================================
    with tab_python:
        st.markdown(
            r"""
            ### Python Codebase Architecture & API Reference

            The application follows a clean modular architecture separating data ingestion, feature generation, ML fitting, explainability, visualization, and telemetry logging.

            ```text
            lake-simulation-timeseries/
            ├── app.py                      # Streamlit UI & Orchestration Layer
            ├── web/                        # GitHub Pages host page & browser entrypoint (stlite)
            ├── scripts/build_site.py       # Assembles the static site for GitHub Pages
            ├── modules/
            │   ├── data_loader.py          # Excel Parsing, Sheet Separation, IQR Audit
            │   ├── feature_engineering.py   # Sine/Cosine Seasonality & Rolling Lags
            │   ├── model.py                # XGBoost Trainer & TimeSeriesSplit CV
            │   ├── explainer.py            # TreeSHAP Explainer Calculation
            │   ├── visualizer.py           # Plotly Interactive Charting Engine
            │   ├── logger.py               # Serialized JSON Telemetry Logger
            │   └── docs_view.py            # Comprehensive Interactive Documentation
            ```

            ---
            #### Module Breakdown & Responsibilities
            * `modules/data_loader.py`: Handles file ingestion (`load_excel`), parses sheets (`separate_sheets`), converts missing value placeholders (`clean_specific_columns`), detects target columns (`detect_target_column`), and flags outliers using IQR (`detect_data_issues`).
            * `modules/feature_engineering.py`: Normalizes time indices, extracts cyclical sine/cosine features for month/hour/day-of-year, and calculates continuous rolling lags.
            * `modules/model.py`: Wraps XGBoost regressor fitting with automated hyperparameter grid search over `TimeSeriesSplit` cross-validation splits.
            * `modules/explainer.py`: Interfaces with `shap.TreeExplainer` to compute exact Shapley value matrices for historical and scenario predictions (in the browser build, XGBoost's built-in TreeSHAP produces the identical values).
            * `modules/visualizer.py`: Generates dark-themed Plotly time-series plots comparing actuals, historical fits, and scenario projections alongside SHAP bar charts.
            * `modules/logger.py`: Writes structured run logs to `logs/simulation_YYYYMMDD_HHMMSS.json` for auditable model tracking.
            """,
            unsafe_allow_html=True,
        )

    # =========================================================================
    # TAB 8: DOCKER & PRODUCTION DEPLOYMENT
    # =========================================================================
    with tab_docker:
        st.markdown(
            r"""
            ### Deployment Guide

            #### Option A: GitHub Pages (Production)
            The public application is a static site: the same Python engine runs directly in the visitor's browser through [stlite](https://github.com/whitphx/stlite) (Streamlit on Pyodide/WebAssembly). No server is involved, and uploaded workbooks never leave the device.

            Every push to the `main` branch runs `.github/workflows/deploy-pages.yml`, which tests the code, builds the site with `scripts/build_site.py`, and publishes it to GitHub Pages. To preview the production build locally:

            ```bash
            python3 scripts/build_site.py --serve
            # Open http://127.0.0.1:8000/
            ```

            ---
            #### Option B: Docker Compose (Self-Hosting)
            Runs the application as a Streamlit server in a container. The files live in `legacy/server/`:

            ```bash
            # 1. Clone the repository
            git clone https://github.com/omidabduli/lake-simulation-timeseries.git
            cd lake-simulation-timeseries

            # 2. Build and launch container in background
            docker compose -f legacy/server/docker-compose.yml up --build -d

            # 3. Access in browser at http://localhost:8501
            ```

            Or with standard Docker commands:
            ```bash
            docker build -f legacy/server/Dockerfile -t lake-forecasting .
            docker run -d -p 8501:8501 -v "$(pwd)/logs:/app/logs" --name lake-app lake-forecasting
            ```

            ---
            #### Option C: macOS / Linux Shell Launcher
            You can also double-click `run.command` or execute it from terminal:
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
        st.markdown("### Frequently Asked Questions & Troubleshooting")

        with st.expander("Why did I get a 'File Structure Error' upon uploading?", expanded=True):
            st.markdown(
                """
                **Cause**: Your uploaded Excel workbook has fewer than 2 sheets or missing column headers.  
                **Fix**: Ensure **Sheet 1** is named Historical (or contains past measurements) and **Sheet 2+** contain future scenarios with identical column headers.
                """
            )

        with st.expander("How does the system know which variable to predict?", expanded=False):
            st.markdown(
                """
                **Answer**: In your scenario sheets (Sheet 2+), leave the column you wish to forecast **completely blank (empty NaN values)** across all rows. The engine scans the sheet, detects the single empty column, and automatically sets it as the simulation target.
                """
            )

        with st.expander("How are missing sentinel values like -999 handled?", expanded=False):
            st.markdown(
                """
                **Answer**: Missing sentinel codes defined in the Data Quality panel are converted to standard `NaN` values and interpolated using continuous forward/linear fill prior to feature engineering.
                """
            )

        with st.expander("Why is TimeSeriesSplit used instead of standard K-Fold CV?", expanded=False):
            st.markdown(
                """
                **Answer**: Standard K-Fold randomly shuffles data rows, causing lookahead bias (training on future dates to predict past dates). `TimeSeriesSplit` strictly enforces forward-only temporal training windows.
                """
            )

        with st.expander("Can I export prediction results for spreadsheet software like Excel?", expanded=False):
            st.markdown(
                """
                **Answer**: Yes! The app features a one-click CSV export formatted specifically for European/German locales (using `;` delimiters and `,` decimal separators) for seamless opening in Microsoft Excel.
                """
            )
