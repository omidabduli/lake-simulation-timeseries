"""
app.py — Universal Time-Series Forecasting.

Streamlit interface: upload a workbook, check the data, train the model and
show scenario forecasts with SHAP explanations. Every scenario runs
automatically after upload.
"""
from __future__ import annotations

import os
import io
import sys
import html
import time
import asyncio
import traceback
import random
import logging

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

# Load demonstration dataset bytes for single-click user download
_EXAMPLE_PATH = os.path.join(os.path.dirname(__file__), "Example", "Lake_Demo_2000_Rows.xlsx")
try:
    with open(_EXAMPLE_PATH, "rb") as _f:
        EXAMPLE_FILE_BYTES = _f.read()
except Exception:
    EXAMPLE_FILE_BYTES = None

from modules.data_loader import (
    clean_specific_columns,
    detect_data_issues,
    detect_target_column,
    load_excel,
    normalize_time_index,
    separate_sheets,
    validate_column_match,
)
from modules.explainer import compute_shap_values
from modules.feature_engineering import engineer_features
from modules.model import train, predict
from modules.visualizer import plot_comparison, plot_shap_bar
from modules.logger import save_simulation_log
from modules.docs_view import render_documentation

# ---------------------------------------------------------------------------
# Runtime: Streamlit server, or in the browser (stlite on Pyodide) for the
# GitHub Pages build. In the browser the script shares one thread with the UI
# message loop, so it must briefly yield before blocking model training for
# queued updates (run header, live console) to be painted.
# ---------------------------------------------------------------------------
IS_BROWSER = sys.platform == "emscripten"
_UI_REFRESH_SECONDS = 0.05 if IS_BROWSER else 0.0

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Universal Time-Series Forecasting",
    layout="wide",
    initial_sidebar_state="auto",
)

# ---------------------------------------------------------------------------
# App modes (persisted in session_state across Streamlit reruns)
#   Basic    — streamlined upload → predict → download flow for beginners
#   Advanced — full data-quality controls, live tuning chart, scorecards
#   seasonal_mode — toggle cyclical month/hour/day-of-year feature engineering
# ---------------------------------------------------------------------------
st.session_state["app_mode"] = "Advanced"
st.session_state.setdefault("seasonal_mode", True)

app_mode = "Advanced"
seasonal_mode = st.session_state.seasonal_mode

# ---------------------------------------------------------------------------
# Design system
# Colors: Warm Neutral (#F9F8F3, #F2EFE7, #FFFFFF), Accent (#1648d8)
# Fonts: Newsreader (Serif), Inter (Sans-Serif), Space Grotesk & JetBrains Mono (Mono)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400..700;1,6..72,400..700&family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap');

    :root {
        /* Surface & Backgrounds */
        --bg-app: #F9F8F3;                /* Main application canvas (warm ivory) */
        --bg-surface: #F2EFE7;            /* Sidebar & secondary background */
        --bg-card: #FFFFFF;               /* Primary card surface */
        --bg-card-alt: #F7F5EE;           /* Highlighted card / accordion container */
        --bg-badge: #191919;              /* Dark pill badge background */

        /* Text & Content */
        --text-primary: #191919;          /* Deep charcoal primary text */
        --text-secondary: #555555;        /* Neutral gray secondary text */
        --text-muted: #707070;            /* Muted detail text */
        --text-disabled: rgba(25, 25, 25, 0.38); /* Disabled-state text/icons */
        --on-badge: #FFFFFF;              /* Text on dark badges */

        /* Primary Brand Accent */
        --primary: #1648d8;
        --primary-hover: #1036aa;
        --primary-light: #E8EEFF;
        --primary-border: #9BB3FF;

        /* Borders & Dividers */
        --border-light: #E6E2D8;
        --border-subtle: #EAE6DD;
        --border-focus: #1648d8;

        /* Status & Checklist Colors */
        --status-complete: #191919;
        --status-pending: #888888;

        /* Shadow & Elevation */
        --shadow-card: 0 2px 8px rgba(0, 0, 0, 0.03);
        --shadow-hover: 0 4px 16px rgba(0, 0, 0, 0.06);

        /* Motion */
        --transition-fast: 120ms ease-out;   /* icon/state toggles */
        --transition-base: 200ms ease;       /* card hover, accordion expand */

        /* Spacing (4px base grid) */
        --space-1: 4px;
        --space-2: 8px;
        --space-3: 12px;
        --space-4: 16px;
        --space-5: 20px;
        --space-6: 24px;
        --space-7: 32px;
        --space-8: 40px;

        /* Legacy variable aliases mapped to modern design tokens */
        --paper: var(--bg-app);
        --ink: var(--text-primary);
        --line: var(--border-light);
        --grid-line: rgba(230, 226, 216, 0.40);
        --blue: var(--primary);
        --sans: 'Inter', system-ui, -apple-system, sans-serif;
        --serif: 'Newsreader', Georgia, serif;
        --mono: 'Space Grotesk', 'JetBrains Mono', monospace;
        --surface: var(--bg-app);
        --surface-container: var(--bg-surface);
        --surface-container-low: var(--bg-card-alt);
        --surface-container-highest: var(--bg-card-alt);
        --on-surface: var(--text-primary);
        --on-surface-variant: var(--text-secondary);
        --outline: var(--border-light);
        --card-bg: var(--bg-card);
        --border: var(--border-light);
        --rounded: 16px;
        --rounded-md: 12px;
        --rounded-lg: 16px;
        --rounded-xl: 20px;
        --card-padding: 24px 32px;
        --stack-gap: 16px;
    }

    /* ── Reset Streamlit defaults ──────────────────────────── */
    html, body, [data-testid="stAppViewContainer"], .main {
        overflow-x: hidden !important;
    }
    /* .stlite-root / overlay roots: in the browser build (stlite) the theme's
       base font is set on these instead of <body>. */
    html, body, .stlite-root, [data-st-overlay-root], [class*="css"] {
        font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
        color: var(--text-primary) !important;
    }
    .stApp {
        background: var(--bg-app) !important;
        color: var(--text-primary) !important;
    }
    [data-testid="stDecoration"] { display: none !important; }
    [data-testid="stToolbar"] { display: flex !important; }
    [data-testid="stHeaderActionElements"] { display: none !important; }
    header[data-testid="stHeader"] {
        height: 2.5rem !important;
        background: transparent !important;
        border: none !important;
    }
    [data-testid="stAppViewContainer"] > .main {
        background: transparent !important;
    }
    .main .block-container {
        max-width: 1240px;
        padding: 3.25rem 2.4rem 4rem !important;
    }

    /* ── Typography Hierarchy (§2 Specification) ───────────── */
    h1, .stMarkdown h1 {
        font-family: 'Newsreader', Georgia, serif !important;
        font-size: clamp(34px, 4.5vw, 44px) !important;
        font-weight: 600 !important;
        line-height: 1.15 !important;
        letter-spacing: -0.02em !important;
        color: var(--text-primary) !important;
    }
    h2, .stMarkdown h2 {
        font-family: 'Newsreader', Georgia, serif !important;
        font-size: clamp(24px, 3.5vw, 28px) !important;
        font-weight: 600 !important;
        line-height: 1.25 !important;
        letter-spacing: -0.01em !important;
        color: var(--text-primary) !important;
    }
    h3, .stMarkdown h3 {
        font-family: 'Newsreader', Georgia, serif !important;
        font-size: clamp(20px, 3vw, 22px) !important;
        font-weight: 600 !important;
        line-height: 1.30 !important;
        letter-spacing: 0em !important;
        color: var(--text-primary) !important;
    }
    h4, .stMarkdown h4 {
        font-family: 'Inter', sans-serif !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        line-height: 1.40 !important;
        letter-spacing: 0em !important;
        color: var(--text-primary) !important;
    }
    .body-lg, p, li, .stMarkdown p, .stMarkdown li {
        font-family: 'Inter', sans-serif !important;
        font-size: 15px !important;
        font-weight: 400 !important;
        line-height: 1.55 !important;
        color: var(--text-secondary) !important;
    }
    .body-sm {
        font-family: 'Inter', sans-serif !important;
        font-size: 13px !important;
        font-weight: 400 !important;
        line-height: 1.50 !important;
        color: var(--text-secondary) !important;
    }
    .label-caps {
        font-family: 'Inter', sans-serif !important;
        font-size: 11px !important;
        font-weight: 600 !important;
        line-height: 1.20 !important;
        letter-spacing: 0.05em !important;
        text-transform: uppercase !important;
        color: var(--text-secondary) !important;
    }
    .mono-data, code {
        font-family: 'Space Grotesk', 'JetBrains Mono', monospace !important;
        font-size: 13px !important;
        font-weight: 400 !important;
        color: var(--text-primary) !important;
    }

    /* ── Dark Pill Badges (§4A Specification) ──────────────── */
    .ui-badge, .target-badge, .doc-meta-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 4px 10px;
        background: var(--bg-badge) !important;
        color: var(--on-badge) !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 11px !important;
        font-weight: 600 !important;
        letter-spacing: 0.02em !important;
        border-radius: 9999px !important;
        border: none !important;
    }

    /* ── Course / Feature Card Container (§4A Specification) ─ */
    .ui-card, .eco-card {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-light) !important;
        border-radius: 16px !important;
        padding: var(--space-6) var(--space-7) !important;
        box-shadow: var(--shadow-card) !important;
        transition: transform var(--transition-base), box-shadow var(--transition-base) !important;
        margin-bottom: var(--space-5) !important;
        position: relative;
        overflow: hidden;
    }
    .ui-card:hover, .eco-card:hover {
        transform: translateY(-2px) !important;
        box-shadow: var(--shadow-hover) !important;
    }
    .ui-card:focus-visible, .eco-card:focus-visible {
        outline: 2px solid var(--border-focus) !important;
        outline-offset: 2px !important;
    }

    /* ── Static Checklist Progress Display (§4B Specification) */
    .ui-list-item {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: var(--space-4) 0;
        border-bottom: 1px solid var(--border-subtle);
    }
    .ui-list-item:last-child {
        border-bottom: none;
    }
    .ui-check-icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 20px;
        height: 20px;
        border-radius: 50%;
        font-size: 11px;
        flex-shrink: 0;
    }
    .ui-check-icon.complete {
        background: var(--status-complete);
        color: #FFFFFF;
    }
    .ui-check-icon.pending {
        border: 2px solid var(--status-pending);
        background: transparent;
        color: transparent;
    }
    .ui-list-title {
        font-family: 'Newsreader', Georgia, serif;
        font-size: 16px;
        font-weight: 500;
        line-height: 1.40;
        color: var(--text-primary);
    }

    /* ── Controls, Buttons & Inputs (§4C Specification) ────── */
    .stButton > button {
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        border-radius: 8px !important;
        padding: 10px 20px !important;
        transition: background var(--transition-fast), border-color var(--transition-fast) !important;
        border: 1px solid var(--border-light) !important;
        background: var(--bg-card) !important;
        color: var(--text-primary) !important;
        box-shadow: none !important;
    }
    .stButton > button:hover {
        background: var(--bg-card-alt) !important;
        border-color: var(--border-light) !important;
        color: var(--text-primary) !important;
    }
    .stButton > button[kind="primary"] {
        background: var(--primary) !important;
        color: #FFFFFF !important;
        border: none !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: var(--primary-hover) !important;
        color: #FFFFFF !important;
    }
    .stButton > button:focus-visible {
        outline: 2px solid var(--border-focus) !important;
        outline-offset: 2px !important;
    }
    .stButton > button:disabled {
        opacity: 0.38 !important;
        cursor: not-allowed !important;
    }

    .stDownloadButton > button {
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        font-size: 13px !important;
        border-radius: 8px !important;
        padding: 10px 18px !important;
        background: var(--bg-card) !important;
        color: var(--primary) !important;
        border: 1px solid var(--primary-border) !important;
        box-shadow: none !important;
        transition: background var(--transition-fast) !important;
    }
    .stDownloadButton > button:hover {
        background: var(--primary-light) !important;
        color: var(--primary-hover) !important;
        border-color: var(--primary) !important;
    }

    /* ── Metric Boxes ─────────────────────────────────────── */
    [data-testid="stMetric"] {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-light) !important;
        border-radius: 16px !important;
        padding: 16px 20px !important;
        box-shadow: var(--shadow-card) !important;
    }
    [data-testid="stMetricValue"] {
        font-family: 'Space Grotesk', 'JetBrains Mono', monospace !important;
        font-size: 24px !important;
        font-weight: 600 !important;
        color: var(--text-primary) !important;
    }
    [data-testid="stMetricLabel"] {
        font-family: 'Inter', sans-serif !important;
        font-size: 11px !important;
        font-weight: 600 !important;
        letter-spacing: 0.05em !important;
        text-transform: uppercase !important;
        color: var(--text-secondary) !important;
    }

    /* ── Expanders ────────────────────────────────────────── */
    .streamlit-expanderHeader {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-light) !important;
        border-radius: 16px !important;
        padding: 16px 20px !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 15px !important;
        font-weight: 600 !important;
        color: var(--text-primary) !important;
    }
    .streamlit-expanderContent {
        background: var(--bg-card-alt) !important;
        border: 1px solid var(--border-light) !important;
        border-top: none !important;
        border-radius: 0 0 16px 16px !important;
        padding: 20px !important;
        margin-top: -1px !important;
    }
        background: var(--paper) !important;
        color: var(--ink) !important;
    }
    [data-testid="stSidebarCollapseButton"] button:hover,
    [data-testid="stExpandSidebarButton"]:hover {
        border-color: var(--blue) !important;
        background: var(--blue) !important;
        color: #fff !important;
    }
    [data-testid="stSidebar"] [data-testid="stSidebarContent"] {
        padding: 0.9rem 1.15rem 1rem !important;
    }
    [data-testid="stSidebar"] hr {
        margin: 0.75rem 0 !important;
        border-color: var(--border) !important;
    }
    [data-testid="stSidebar"] .stCaption {
        color: #68675f !important;
        margin-top: -0.25rem;
    }
    .brand-lockup {
        display: flex;
        align-items: center;
        gap: 0.85rem;
        padding: 0.35rem 0 1.2rem;
    }
    .brand-mark {
        width: 18px;
        height: 18px;
        flex: 0 0 18px;
        border: 2px solid var(--ink);
        border-radius: 50%;
        background: transparent;
        position: relative;
    }
    .brand-mark::after {
        content: "";
        position: absolute;
        width: 7px;
        height: 7px;
        right: -6px;
        top: -5px;
        border-radius: 50%;
        background: var(--blue);
    }
    .brand-copy h1 {
        margin: 0 !important;
        color: var(--ink) !important;
        font-family: var(--sans) !important;
        font-size: 17px !important;
        line-height: 1.15 !important;
        font-weight: 600 !important;
        letter-spacing: -0.025em !important;
    }
    .brand-copy p {
        margin: 0.25rem 0 0 !important;
        color: #66645d !important;
        font-family: var(--mono) !important;
        font-size: 9px !important;
        font-weight: 400 !important;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .sidebar-label {
        margin: 0.2rem 0 0.55rem !important;
        color: var(--ink) !important;
        font-family: var(--mono) !important;
        font-size: 10px !important;
        font-weight: 400 !important;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .sidebar-note {
        display: flex;
        gap: 0.65rem;
        align-items: flex-start;
        margin-top: 1rem;
        padding: 0.85rem 0.9rem;
        border: 1px solid var(--ink);
        border-radius: 0;
        background: var(--paper);
    }
    .sidebar-note .material-symbols-outlined {
        color: var(--blue) !important;
        font-size: 18px;
    }
    .sidebar-note p {
        margin: 0 !important;
        color: #4f4e48 !important;
        font-size: 12px !important;
        line-height: 1.5 !important;
    }
    .sidebar-signoff {
        margin-top: 1.35rem;
        color: #6f6d65 !important;
        font-family: var(--mono) !important;
        font-size: 10px !important;
        font-weight: 400 !important;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    .sidebar-doc-link {
        margin-top: 2px !important;
    }
    .sidebar-doc-link button {
        background: transparent !important;
        border: none !important;
        color: var(--blue) !important;
        font-family: var(--mono) !important;
        font-size: 10px !important;
        font-weight: 500 !important;
        letter-spacing: 0.05em !important;
        text-transform: uppercase !important;
        text-decoration: underline !important;
        padding: 0 !important;
        margin: 0 !important;
        min-height: auto !important;
        height: auto !important;
        line-height: 1.6 !important;
        box-shadow: none !important;
    }
    .sidebar-doc-link button:hover {
        color: #0b2d94 !important;
        background: transparent !important;
    }

    /* Upload and controls */
    [data-testid="stFileUploader"] > label {
        display: none;
    }
    [data-testid="stFileUploaderDropzone"] {
        min-height: 112px;
        padding: 1rem !important;
        background: var(--paper) !important;
        border: 1px dashed var(--ink) !important;
        border-radius: 0 !important;
        transition: border-color 160ms ease, background 160ms ease, transform 160ms ease;
    }
    [data-testid="stFileUploaderDropzone"]:hover {
        background: #e8ecfa !important;
        border-color: var(--blue) !important;
        transform: none;
    }
    [data-testid="stFileUploaderDropzone"] button {
        border-radius: 0 !important;
        border: 1px solid var(--blue) !important;
        background: var(--blue) !important;
        color: #fff !important;
        font-family: var(--mono) !important;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] button,
    [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] button * {
        color: #fff !important;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] {
        display: grid !important;
        grid-template-columns: 1fr 1fr;
        gap: 0.35rem;
        padding: 0.3rem;
        border: 1px solid var(--ink);
        border-radius: 0;
        background: var(--paper);
    }
    [data-testid="stSidebar"] div[role="radiogroup"] label {
        justify-content: center;
        min-height: 34px;
        border-radius: 0;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] label,
    [data-testid="stSidebar"] div[role="radiogroup"] label * {
        color: var(--ink) !important;
    }
    [data-testid="stSidebar"] [data-testid="stCheckbox"] {
        padding: 0.6rem 0.75rem;
        border: 1px solid var(--ink);
        border-radius: 0;
        background: var(--paper);
    }

    /* Main hero */
    .hero-shell {
        position: relative;
        overflow: hidden;
        min-height: 360px;
        padding: clamp(2rem, 5vw, 4.5rem);
        border: 1px solid var(--line);
        border-radius: 0;
        background:
            linear-gradient(var(--line) 1px, transparent 1px),
            linear-gradient(90deg, var(--line) 1px, transparent 1px),
            var(--paper) !important;
        background-size: 28px 28px !important;
        background-position: -1px -1px !important;
        box-shadow: none;
    }
    .signal {
        width: 170px;
        height: 170px;
        aspect-ratio: 1;
        padding: 0;
        border: 0;
        background: var(--blue) !important;
        border-radius: 50% !important;
        position: absolute;
        right: 25px !important;
        top: 25px !important;
        left: auto !important;
        z-index: 5;
        cursor: grab;
        touch-action: none;
        user-select: none;
        -webkit-user-select: none;
        mix-blend-mode: multiply;
        box-shadow: none;
        will-change: transform;
        outline-offset: 5px;
    }
    .signal:active, .signal.is-dragging {
        cursor: grabbing;
        box-shadow: none;
    }
    .signal:focus-visible {
        outline: 2px solid var(--ink);
    }
    .signal::before, .signal::after {
        content: "";
        position: absolute;
        inset: 22%;
        border: 1px solid rgba(255, 255, 255, 0.85);
        border-radius: 50%;
        pointer-events: none;
    }
    .signal::after {
        inset: 43%;
        background: var(--paper) !important;
        border: 0;
    }
    .hero-kicker {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        margin-bottom: 1.1rem;
        color: var(--ink) !important;
        font-family: var(--mono) !important;
        font-size: 11px !important;
        font-weight: 400 !important;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .hero-kicker::before {
        content: "";
        width: 24px;
        height: 1px;
        border-radius: 0;
        background: var(--ink);
        box-shadow: none;
    }
    .stMarkdown .hero-shell h1 {
        position: relative;
        z-index: 1;
        max-width: 760px;
        margin: 0 !important;
        color: var(--ink) !important;
        font-family: var(--serif) !important;
        font-size: clamp(48px, 7vw, 92px) !important;
        font-weight: 400 !important;
        line-height: 0.9 !important;
        letter-spacing: -0.06em !important;
    }
    .stMarkdown .hero-shell h1 > span {
        color: var(--ink) !important;
    }
    .stMarkdown .hero-shell h1 > span > span {
        color: var(--blue) !important;
        font-family: var(--serif) !important;
        font-weight: 400 !important;
    }
    .hero-shell > p {
        position: relative;
        z-index: 1;
        max-width: 620px;
        margin: 1.3rem 0 0 !important;
        color: #4d4c46 !important;
        font-family: var(--sans) !important;
        font-size: 17px !important;
        line-height: 1.65 !important;
    }
    .hero-flow {
        position: relative;
        z-index: 1;
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 0.55rem;
        margin-top: 2rem;
    }
    .hero-flow span {
        padding: 0.5rem 0.75rem;
        border: 1px solid var(--ink);
        border-radius: 0;
        background: var(--paper);
        color: var(--ink) !important;
        font-family: var(--mono) !important;
        font-size: 10px !important;
        font-weight: 400;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }
    .hero-flow b {
        color: var(--blue) !important;
        font-size: 13px;
    }
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 0;
        margin-top: 0;
        border: 1px solid var(--line);
        border-top: 0;
    }
    .feature-tile {
        min-height: 148px;
        padding: 1.25rem;
        border: 0;
        border-radius: 0;
        background: var(--paper);
    }
    .feature-tile + .feature-tile {
        border-left: 1px solid var(--line);
    }
    .feature-tile .material-symbols-outlined {
        color: var(--blue) !important;
        font-size: 23px;
    }
    .feature-tile h3 {
        margin: 0.75rem 0 0.4rem !important;
        color: var(--ink) !important;
        font-family: var(--serif) !important;
        font-size: 22px !important;
        font-weight: 400 !important;
        line-height: 1.3 !important;
    }
    .feature-tile p {
        margin: 0 !important;
        color: #65635c !important;
        font-size: 12px !important;
        line-height: 1.55 !important;
    }

    /* Dashboard surfaces */
    .eco-card,
    [data-testid="stMetric"],
    .footer-bar {
        border-color: var(--border) !important;
        background: var(--card-bg) !important;
        border-radius: 0 !important;
        box-shadow: none;
    }
    .eco-card::before { display: none; }
    [data-testid="stMetric"] {
        min-height: 112px;
        padding: 1.1rem 1.15rem !important;
    }
    [data-testid="stMetricValue"] {
        color: var(--ink) !important;
        font-family: var(--serif) !important;
        font-size: 26px !important;
        font-weight: 400 !important;
    }
    [data-testid="stMetricValue"] > div {
        overflow: visible !important;
        text-overflow: clip !important;
        white-space: normal !important;
        overflow-wrap: anywhere;
        line-height: 1.05 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #5a5851 !important;
        font-family: var(--mono) !important;
        font-size: 10px !important;
        font-weight: 500 !important;
        letter-spacing: 0.05em !important;
    }
    [data-testid="stAlert"] {
        border: 1px solid var(--line) !important;
        border-left: 4px solid var(--blue) !important;
        border-radius: 0 !important;
        background: #ece9df !important;
        color: var(--ink) !important;
        box-shadow: none !important;
    }
    [data-testid="stAlert"] * {
        color: var(--ink) !important;
    }
    [data-testid="stExpander"] {
        border: 1px solid var(--line) !important;
        border-radius: 0 !important;
        background: var(--paper) !important;
        box-shadow: none !important;
    }
    [data-testid="stExpander"] summary {
        min-height: 48px;
        font-family: var(--mono) !important;
        font-size: 11px !important;
        letter-spacing: 0.02em;
    }
    .section-heading {
        display: grid;
        grid-template-columns: 48px minmax(0, 1fr);
        gap: 1rem;
        align-items: start;
        margin: 2rem 0 1rem;
        padding: 1.1rem 0;
        border-top: 1px solid var(--ink);
        border-bottom: 1px solid var(--line);
    }
    .section-heading__number {
        color: var(--blue) !important;
        font-family: var(--mono) !important;
        font-size: 10px !important;
        letter-spacing: 0.08em;
    }
    .section-heading h2 {
        margin: 0 !important;
        color: var(--ink) !important;
        font-family: var(--serif) !important;
        font-size: 34px !important;
        font-weight: 400 !important;
        line-height: 1 !important;
        letter-spacing: -0.035em !important;
    }
    .section-heading p {
        margin: 0.4rem 0 0 !important;
        max-width: 720px;
        color: #65635c !important;
        font-size: 13px !important;
        line-height: 1.5 !important;
    }
    .workflow-summary {
        display: grid;
        grid-template-columns: 145px repeat(3, minmax(0, 1fr));
        margin-bottom: 1.5rem;
        border: 1px solid var(--ink);
        background: var(--paper);
    }
    .workflow-summary__label,
    .workflow-summary__item {
        min-height: 82px;
        padding: 1rem;
    }
    .workflow-summary__label {
        display: flex;
        align-items: center;
        border-right: 1px solid var(--ink);
        color: var(--blue) !important;
        font-family: var(--mono) !important;
        font-size: 10px !important;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }
    .workflow-summary__item + .workflow-summary__item {
        border-left: 1px solid var(--line);
    }
    .workflow-summary__item span {
        display: block;
        margin-bottom: 0.45rem;
        color: #77746c !important;
        font-family: var(--mono) !important;
        font-size: 9px !important;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }
    .workflow-summary__item strong {
        color: var(--ink) !important;
        font-family: var(--serif) !important;
        font-size: 22px;
        font-weight: 400;
    }
    .issue-banner {
        display: grid;
        grid-template-columns: 110px minmax(0, 1fr);
        margin: 1.25rem 0;
        border: 1px solid var(--ink);
        background: #e9e5da;
    }
    .issue-banner__count {
        display: flex;
        flex-direction: column;
        justify-content: center;
        padding: 1rem;
        border-right: 1px solid var(--ink);
        color: var(--blue) !important;
        font-family: var(--serif) !important;
        font-size: 34px !important;
        line-height: 0.9;
    }
    .issue-banner__count small {
        margin-top: 0.45rem;
        color: #65635c !important;
        font-family: var(--mono) !important;
        font-size: 8px !important;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .issue-banner__copy {
        padding: 1rem 1.25rem;
    }
    .issue-banner__copy strong {
        display: block;
        color: var(--ink) !important;
        font-size: 14px;
    }
    .issue-banner__copy p {
        margin: 0.3rem 0 0 !important;
        color: #65635c !important;
        font-size: 12px !important;
    }
    .selection-summary {
        margin: 1rem 0;
        padding: 1rem 1.1rem;
        border: 1px solid var(--line);
        background: var(--paper);
    }
    .selection-summary strong {
        color: var(--ink) !important;
    }
    .selection-summary p {
        margin: 0.25rem 0 0 !important;
        color: #65635c !important;
        font-size: 12px !important;
    }
    .tuning-readout {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        margin-top: 1rem;
        border: 1px solid var(--ink);
        background: var(--paper);
    }
    .tuning-readout div {
        min-width: 0;
        padding: 0.9rem 1rem;
    }
    .tuning-readout div + div {
        border-left: 1px solid var(--line);
    }
    .tuning-readout span {
        display: block;
        margin-bottom: 0.35rem;
        color: #706e66 !important;
        font-family: var(--mono) !important;
        font-size: 8px !important;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }
    .tuning-readout strong {
        color: var(--ink) !important;
        font-size: 13px;
    }
    .score-stack {
        display: grid;
        gap: 0;
        border: 1px solid var(--ink);
    }
    .score-stack__header {
        padding: 1rem;
        border-bottom: 1px solid var(--ink);
        background: var(--ink);
        color: var(--paper) !important;
        font-family: var(--mono) !important;
        font-size: 10px !important;
        letter-spacing: 0.07em;
        text-transform: uppercase;
    }
    .score-stack__item {
        padding: 1rem;
        background: var(--paper);
    }
    .score-stack__item + .score-stack__item {
        border-top: 1px solid var(--line);
    }
    .score-stack__item span {
        display: block;
        margin-bottom: 0.3rem;
        color: #706e66 !important;
        font-family: var(--mono) !important;
        font-size: 9px !important;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }
    .score-stack__item strong {
        display: block;
        color: var(--ink) !important;
        font-family: var(--serif) !important;
        font-size: 24px;
        font-weight: 400;
        line-height: 1.05;
        overflow-wrap: anywhere;
    }
    .score-stack__item--target strong {
        color: var(--blue) !important;
        font-family: var(--sans) !important;
        font-size: 14px;
        font-weight: 650;
        line-height: 1.35;
    }
    .model-facts {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        margin-top: 2rem;
        border: 1px solid var(--ink);
        background: var(--paper);
    }
    .model-fact {
        min-width: 0;
        padding: 1.15rem;
    }
    .model-fact + .model-fact {
        border-left: 1px solid var(--line);
    }
    .model-fact:nth-child(4) {
        border-left: 0;
    }
    .model-fact:nth-child(n + 4) {
        border-top: 1px solid var(--line);
    }
    .model-fact span {
        display: block;
        margin-bottom: 0.45rem;
        color: #706e66 !important;
        font-family: var(--mono) !important;
        font-size: 9px !important;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }
    .model-fact strong {
        color: var(--ink) !important;
        font-size: 13px;
        overflow-wrap: anywhere;
    }
    .model-status {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        color: #173b27 !important;
    }
    .model-status::before {
        content: "";
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #16845b;
    }
    /* ── Streamlit Tabs Comprehensive Fix ───────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        border-bottom: 1px solid var(--border-light, #E6E2D8) !important;
        background: transparent !important;
        background-color: transparent !important;
        gap: 1.5rem !important;
        padding-bottom: 0 !important;
        margin-bottom: 1.5rem !important;
    }
    .stTabs button[data-baseweb="tab"],
    .stTabs [data-baseweb="tab"],
    .stTabs div[role="tab"] {
        font-family: 'Inter', sans-serif !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        color: var(--text-secondary, #555555) !important;
        background: transparent !important;
        background-color: transparent !important;
        border: none !important;
        border-bottom: 2px solid transparent !important;
        border-radius: 0 !important;
        padding: 0.6rem 0.25rem !important;
        outline: none !important;
        box-shadow: none !important;
    }
    .stTabs button[data-baseweb="tab"] *,
    .stTabs [data-baseweb="tab"] *,
    .stTabs div[role="tab"] * {
        font-family: 'Inter', sans-serif !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        color: var(--text-secondary, #555555) !important;
        background: transparent !important;
        background-color: transparent !important;
    }
    .stTabs button[data-baseweb="tab"]:hover,
    .stTabs button[data-baseweb="tab"]:hover *,
    .stTabs [data-baseweb="tab"]:hover,
    .stTabs [data-baseweb="tab"]:hover * {
        color: var(--text-primary, #191919) !important;
        background: transparent !important;
        background-color: transparent !important;
    }
    .stTabs button[aria-selected="true"],
    .stTabs button[data-baseweb="tab"][aria-selected="true"],
    .stTabs [data-baseweb="tab"][aria-selected="true"],
    .stTabs div[role="tab"][aria-selected="true"] {
        background: transparent !important;
        background-color: transparent !important;
        border-bottom: 2px solid var(--primary, #1648d8) !important;
        border-radius: 0 !important;
        outline: none !important;
        box-shadow: none !important;
    }
    .stTabs button[aria-selected="true"] *,
    .stTabs button[data-baseweb="tab"][aria-selected="true"] *,
    .stTabs [data-baseweb="tab"][aria-selected="true"] *,
    .stTabs div[role="tab"][aria-selected="true"] * {
        color: var(--primary, #1648d8) !important;
        font-weight: 600 !important;
        background: transparent !important;
        background-color: transparent !important;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: var(--primary, #1648d8) !important;
    }
    .stTabs [data-baseweb="tab-border"] {
        background-color: var(--border-light, #E6E2D8) !important;
    }
    .stDownloadButton > button {
        border-radius: 0 !important;
        border-color: var(--blue) !important;
        background: var(--blue) !important;
        color: #fff !important;
        font-family: var(--mono) !important;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }
    [data-testid="stPlotlyChart"] {
        overflow: hidden;
        padding: 0.25rem;
        border: 1px solid var(--border);
        border-radius: 0;
        background: var(--card-bg);
        box-shadow: none;
    }
    [data-testid="stProgress"] > div > div {
        background: var(--blue) !important;
        border-radius: 0 !important;
    }
    [data-testid="stProgress"] > div {
        height: 6px !important;
        border-radius: 0 !important;
        background: #ddd8cc !important;
    }
    .validation-strip {
        display: grid;
        grid-template-columns: 42px minmax(0, 1fr);
        gap: 0.9rem;
        align-items: start;
        margin-bottom: 1rem;
        padding: 0.9rem 1rem;
        border: 1px solid var(--line);
        border-left: 4px solid var(--blue);
        background: var(--paper);
    }
    .validation-strip__index {
        padding-top: 0.1rem;
        color: var(--blue) !important;
        font-family: var(--mono) !important;
        font-size: 10px !important;
        letter-spacing: 0.08em;
    }
    .validation-strip strong {
        display: block;
        margin-bottom: 0.2rem;
        color: var(--ink) !important;
        font-family: var(--sans) !important;
        font-size: 13px;
        font-weight: 650;
    }
    .validation-strip p {
        margin: 0 !important;
        color: #65635c !important;
        font-size: 12px !important;
        line-height: 1.5 !important;
    }
    .run-stage {
        display: grid;
        grid-template-columns: 180px minmax(0, 1fr);
        margin-top: 0.5rem;
        border: 1px solid var(--ink);
        background: var(--paper);
    }
    .run-stage__kicker {
        display: flex;
        align-items: center;
        padding: 1.1rem;
        border-right: 1px solid var(--ink);
        color: var(--blue) !important;
        font-family: var(--mono) !important;
        font-size: 10px !important;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .run-stage__copy {
        padding: 1rem 1.25rem;
    }
    .run-stage__copy h2 {
        margin: 0 !important;
        color: var(--ink) !important;
        font-family: var(--serif) !important;
        font-size: 30px !important;
        font-weight: 400 !important;
        line-height: 1.05 !important;
        letter-spacing: -0.035em !important;
    }
    .run-stage__copy p {
        margin: 0.35rem 0 0 !important;
        color: #65635c !important;
        font-size: 12px !important;
    }
    .run-status-row {
        display: grid;
        grid-template-columns: 72px minmax(0, 1fr) auto;
        gap: 1rem;
        align-items: center;
        padding: 0.75rem 0.9rem;
        border: 1px solid var(--ink);
        border-top: 0;
        background: var(--paper);
        font-family: var(--mono) !important;
        font-size: 10px !important;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }
    .run-status-row span {
        color: var(--blue) !important;
    }
    .run-status-row strong {
        overflow: hidden;
        color: var(--ink) !important;
        font-weight: 500;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    .run-status-row em {
        color: #6b6961 !important;
        font-style: normal;
    }
    .training-console {
        position: relative;
        overflow: hidden;
        margin: 0.8rem 0;
        border: 1px solid var(--ink);
        background: var(--ink);
        color: var(--paper);
    }
    .training-console::after {
        content: "";
        position: absolute;
        inset: 0;
        pointer-events: none;
        background: linear-gradient(
            110deg,
            transparent 0%,
            transparent 42%,
            rgba(48, 91, 231, 0.2) 49%,
            transparent 56%,
            transparent 100%
        );
        transform: translateX(-100%);
        animation: training-scan 2.4s linear infinite;
    }
    @keyframes training-scan {
        to { transform: translateX(100%); }
    }
    .training-console__top {
        display: flex;
        justify-content: space-between;
        gap: 1rem;
        padding: 0.85rem 1rem;
        border-bottom: 1px solid #464641;
        color: #c9c5ba !important;
        font-family: var(--mono) !important;
        font-size: 9px !important;
        letter-spacing: 0.07em;
        text-transform: uppercase;
    }
    .training-console__top b {
        color: #7e9cff !important;
        font-weight: 500;
    }
    .training-console__body {
        display: grid;
        grid-template-columns: minmax(0, 1.35fr) minmax(260px, 0.65fr);
    }
    .training-console__activity {
        padding: 1.25rem;
        border-right: 1px solid #464641;
    }
    .training-console__activity h3 {
        margin: 0 !important;
        color: #f2efe7 !important;
        font-family: var(--serif) !important;
        font-size: 30px !important;
        font-weight: 400 !important;
        letter-spacing: -0.03em !important;
    }
    .training-console__activity p {
        margin: 0.35rem 0 1rem !important;
        color: #a9a69d !important;
        font-size: 12px !important;
    }
    .training-bars {
        display: grid;
        grid-template-columns: repeat(15, 1fr);
        gap: 4px;
        height: 52px;
        align-items: end;
    }
    .training-bars span {
        min-height: 7px;
        border: 1px solid #4e4d48;
        background: #292925;
    }
    .training-bars span.done {
        border-color: #305be7;
        background: #305be7;
    }
    .training-bars span.active {
        border-color: #8aa3ff;
        background: #8aa3ff;
        animation: training-pulse 760ms ease-in-out infinite alternate;
    }
    @keyframes training-pulse {
        from { height: 35%; opacity: 0.55; }
        to { height: 100%; opacity: 1; }
    }
    .training-console__stats {
        display: grid;
        grid-template-columns: 1fr 1fr;
    }
    .training-stat {
        padding: 1rem;
        border-bottom: 1px solid #464641;
    }
    .training-stat:nth-child(odd) {
        border-right: 1px solid #464641;
    }
    .training-stat span {
        display: block;
        margin-bottom: 0.35rem;
        color: #8f8c83 !important;
        font-family: var(--mono) !important;
        font-size: 8px !important;
        letter-spacing: 0.07em;
        text-transform: uppercase;
    }
    .training-stat strong {
        color: #f2efe7 !important;
        font-family: var(--mono) !important;
        font-size: 14px;
        font-weight: 500;
    }
    .training-stat strong.best {
        color: #8aa3ff !important;
    }
    .result-overview {
        display: flex;
        align-items: flex-end;
        justify-content: space-between;
        gap: 1.5rem;
        margin-bottom: 1.2rem;
    }
    .result-overview h1 {
        margin: 0 !important;
        color: var(--ink) !important;
        font-family: var(--serif) !important;
        font-size: 52px !important;
        font-weight: 400 !important;
        line-height: 0.95 !important;
        letter-spacing: -0.05em !important;
    }
    .result-overview p {
        max-width: 490px;
        margin: 0 !important;
        color: #626159 !important;
        font-size: 13px !important;
        text-align: right;
    }

    @media (max-width: 900px) {
        .main .block-container {
            padding: 2.5rem 1rem 3rem !important;
        }
        .feature-grid {
            grid-template-columns: 1fr;
        }
        .hero-shell {
            min-height: auto;
            padding: 2rem 1.4rem;
            border-radius: 0;
        }
        .hero-shell h1 {
            font-size: 52px !important;
        }
        .hero-shell::after {
            display: none;
        }
        .footer-bar {
            grid-template-columns: 1fr 1fr !important;
        }
        .result-overview {
            display: block;
        }
        .result-overview p {
            margin-top: 0.65rem !important;
            text-align: left;
        }
        .run-stage {
            grid-template-columns: 1fr;
        }
        .run-stage__kicker {
            padding-bottom: 0;
            border-right: 0;
        }
        .run-status-row {
            grid-template-columns: 58px minmax(0, 1fr);
        }
        .run-status-row em {
            display: none;
        }
        .workflow-summary {
            grid-template-columns: 1fr 1fr;
        }
        .workflow-summary__label {
            grid-column: 1 / -1;
            min-height: auto;
            border-right: 0;
            border-bottom: 1px solid var(--ink);
        }
        .workflow-summary__item + .workflow-summary__item {
            border-left: 0;
        }
        .model-facts {
            grid-template-columns: 1fr 1fr;
        }
        .model-fact:nth-child(odd) {
            border-left: 0;
        }
        .model-fact:nth-child(even) {
            border-left: 1px solid var(--line);
        }
        .model-fact:nth-child(n + 3) {
            border-top: 1px solid var(--line);
        }
        .training-console__body {
            grid-template-columns: 1fr;
        }
        .training-console__activity {
            border-right: 0;
            border-bottom: 1px solid #464641;
        }
        .tuning-readout {
            grid-template-columns: 1fr;
        }
        .tuning-readout div + div {
            border-top: 1px solid var(--line);
            border-left: 0;
        }
    }
    @media (min-width: 901px) {
        .hero-shell {
            padding-right: 300px;
        }
        .hero-shell::after {
            width: 250px;
            height: 250px;
            right: 34px;
            top: 34px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar — branding + file upload
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="brand-lockup">
            <div class="brand-mark" aria-hidden="true"></div>
            <div class="brand-copy">
                <h1>Time-Series Forecasting</h1>
                <p>Universal scenario engine</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<p class="sidebar-label">Dataset</p>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Upload Dataset",
        type=["xls", "xlsx"],
        help="Excel file: Sheet 1 = Historical, Sheet 2+ = Scenarios (one column empty).",
    )
    if EXAMPLE_FILE_BYTES:
        st.download_button(
            label="Download Example File",
            data=EXAMPLE_FILE_BYTES,
            file_name="Time_Series_Forecasting_Demo_2000_Rows.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            help="Download a sample Excel dataset to see how your workbook should be formatted.",
        )
    st.markdown(
        """
        <div class="sidebar-note">
            <span class="material-symbols-outlined">table_view</span>
            <p>Use one historical sheet followed by one or more scenario sheets. Leave the prediction target empty in each scenario.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown('<p class="sidebar-label">Model options</p>', unsafe_allow_html=True)
    st.checkbox(
        "Seasonal effects",
        key="seasonal_mode",
        help="Use cyclical month / hour / day-of-year features to capture calendar seasonality "
             "(recommended for seasonal, financial, weather, demand, or operational data). Turn OFF for non-seasonal data.",
    )
    show_doc = st.session_state.get("show_doc", False)
    st.markdown(
        """
        <div class="sidebar-signoff" style="line-height:1.6;">
            <div>v2.0.0 · XGBoost · SHAP · Time-series CV</div>
            <div>By <a href="https://github.com/omidabduli" target="_blank" style="color:var(--blue);text-decoration:underline;">Omid Abduli</a> · <a href="https://github.com/omidabduli/lake-simulation-timeseries" target="_blank" style="color:var(--ink);text-decoration:underline;">GitHub ↗</a></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    doc_link_label = "← Back to Workspace" if show_doc else "Documentation"
    st.markdown('<div class="sidebar-doc-link">', unsafe_allow_html=True)
    if st.button(doc_link_label, key="sidebar_doc_toggle"):
        st.session_state["show_doc"] = not show_doc
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# _NoOp: a stand-in for Streamlit placeholders that silently absorbs every
# rendering call (line_chart / markdown / progress / empty). It lets the shared
# (Advanced) optimization-UI body run unchanged in Basic mode, where the live
# racing chart is intentionally hidden — calls just resolve to no-ops.
# ---------------------------------------------------------------------------
class _NoOp:
    def __getattr__(self, _name):
        return lambda *args, **kwargs: None


def _display_field_name(field_name: str) -> str:
    """Convert machine-oriented column names into compact readable labels."""
    tokens = str(field_name).replace("-", "_").split("_")
    units = {
        "mg": "mg",
        "ug": "µg",
        "l": "L",
        "c": "°C",
        "ms": "m/s",
        "ph": "pH",
    }
    words = [units.get(token.lower(), token.capitalize()) for token in tokens if token]
    label = " ".join(words)
    return label.replace("mg L", "mg/L").replace("µg L", "µg/L")



# ---------------------------------------------------------------------------
# Helper: run full pipeline for one scenario (cached)
# ---------------------------------------------------------------------------
async def _run_scenario(
    historical_raw: pd.DataFrame,
    scenario_raw: pd.DataFrame,
    scenario_name: str,
    *,
    seasonal: bool = True,
    show_optimization_ui: bool = True,
    status_placeholder=None,
    progress_placeholder=None,
    scenario_index: int = 0,
    scenario_total: int = 1,
    search_iterations: int = 15,
    cv_folds: int = 4,
    depth_range: tuple[int, int] = (2, 8),
    rolling_windows: tuple[int, ...] = (3, 7),
) -> dict | str:
    """
    Run the full ML pipeline for a single scenario.

    This function orchestrates the entire simulation process for a given scenario:
    1. Validates the structural integrity of the input data.
    2. Identifies the target variable (the missing column in the scenario).
    3. Normalizes time indices for consistent time-series alignment.
    4. Engineers lag and rolling features for the historical data.
    5. Trains an XGBoost model on the historical feature set.
    6. Generates predictions to fill the missing values in the scenario.
    7. Computes model performance metrics (R², RMSE) and SHAP values for explainability.

    Keyword-only arguments:
        seasonal : bool
            Passed to engineer_features — toggles cyclical seasonal features.
        show_optimization_ui : bool
            When True (Advanced), render the live per-iteration hyperparameter
            racing chart + stats card. When False (Basic), run the same search
            silently behind a single spinner with no artificial delay.

    The function is async only so the UI can repaint between optimization
    candidates in the browser build; the calculations are unchanged.

    Returns:
        A dictionary containing the simulation results, or a string describing an error.
    """
    # 1. Structural validation: Ensure scenario columns match historical columns (except for the target)
    try:
        validate_column_match(historical_raw, scenario_raw)
    except ValueError as e:
        return str(e)

    # 2. Target identification: Find the single completely empty column in the scenario
    try:
        target_col = detect_target_column(scenario_raw)
    except ValueError as e:
        return str(e)

    # 3. Time normalization: Set Datetime index to ensure strict chronological ordering
    try:
        hist_df = normalize_time_index(historical_raw)
        scen_df = normalize_time_index(scenario_raw)
    except ValueError as e:
        return str(e)

    # Each validation fold needs at least two observations for R² to be
    # defined. Guard the chosen validation rigor against short histories.
    minimum_rows = 2 * (cv_folds + 1)
    if len(hist_df) < minimum_rows:
        return (
            f"Historical data has only {len(hist_df)} rows. "
            f"At least {minimum_rows} rows are required for "
            f"{cv_folds}-fold time-series validation."
        )

    # 4. Feature engineering: Extract temporal features, lags, and rolling averages
    time_origin = hist_df.index.min()
    X_hist, y_hist = engineer_features(
        hist_df,
        target_col=target_col,
        seasonal=seasonal,
        rolling_windows=rolling_windows,
        time_origin=time_origin,
    )
    X_scen, _ = engineer_features(
        scen_df,
        target_col=target_col,
        seasonal=seasonal,
        rolling_windows=rolling_windows,
        time_origin=time_origin,
    )

    # Align scenario features with historical features (fill missing with NaN to prevent bias, XGBoost handles NaNs natively)
    X_scen = X_scen.reindex(columns=X_hist.columns, fill_value=np.nan)

    # 5. Model Optimization & Training
    
    training_placeholder = st.empty() if show_optimization_ui else _NoOp()
    await asyncio.sleep(_UI_REFRESH_SECONDS)

    best_cv_r2 = -float("inf")
    best_cv_rmse = float("inf")
    best_params = {}
    history_r2 = []
    
    n_iterations = search_iterations
    t0 = time.time()
    
    # Fix random seed for reproducibility. A private generator yields the same
    # sequence as random.seed(42), but other threads cannot shift it: on a
    # Streamlit server, WebSocket keep-alive pings draw from the global
    # `random` state, which made the candidate search differ between runs.
    rng = random.Random(42)
    
    try:
        # TimeSeriesSplit ensures we never train on future data to predict the past
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        
        for i in range(n_iterations):
            # Generate random hyperparameters
            lr = rng.uniform(0.01, 0.2)
            max_depth = rng.randint(*depth_range)
            n_estimators = rng.randint(100, 500)
            
            params = {
                "learning_rate": lr,
                "max_depth": max_depth,
                "n_estimators": n_estimators,
            }
            
            fold_r2s = []
            fold_rmses = []
            
            # Cross-validate the current parameter set
            for train_idx, val_idx in tscv.split(X_hist):
                X_t, X_v = X_hist.iloc[train_idx], X_hist.iloc[val_idx]
                y_t, y_v = y_hist.iloc[train_idx], y_hist.iloc[val_idx]
                
                fold_model = train(X_t, y_t, params=params)
                y_v_pred = predict(fold_model, X_v)
                
                fold_r2s.append(r2_score(y_v, y_v_pred))
                fold_rmses.append(np.sqrt(mean_squared_error(y_v, y_v_pred)))
            
            current_cv_r2 = np.mean(fold_r2s)
            current_cv_rmse = np.mean(fold_rmses)

            if not np.isfinite(current_cv_r2) or not np.isfinite(current_cv_rmse):
                continue
            
            # Track best cross-validated model parameters
            if current_cv_r2 > best_cv_r2:
                best_cv_r2 = current_cv_r2
                best_cv_rmse = current_cv_rmse
                best_params = params
                
            history_r2.append({"Iteration": i + 1, "Current CV R²": current_cv_r2, "Best CV R²": best_cv_r2})
            safe_scenario_name = html.escape(scenario_name.removeprefix("Scenario: "))
            bars = "".join(
                f'<span class="{"active" if bar_idx == i else "done" if bar_idx < i else ""}" '
                f'style="height:{22 + ((bar_idx * 17) % 68)}%"></span>'
                for bar_idx in range(n_iterations)
            )

            training_placeholder.markdown(
                f"""
                <div class="training-console">
                    <div class="training-console__top">
                        <span>Forecasting model lab / {safe_scenario_name}</span>
                        <b>Live optimization</b>
                    </div>
                    <div class="training-console__body">
                        <div class="training-console__activity">
                            <h3>Training forecast engine</h3>
                            <p>Testing candidate {i + 1} of {n_iterations} across {cv_folds} forward-only validation folds.</p>
                            <div class="training-bars">{bars}</div>
                        </div>
                        <div class="training-console__stats">
                            <div class="training-stat"><span>Iteration</span><strong>{i + 1:02d} / {n_iterations:02d}</strong></div>
                            <div class="training-stat"><span>Current CV R²</span><strong>{current_cv_r2:.4f}</strong></div>
                            <div class="training-stat"><span>Best CV R²</span><strong class="best">{best_cv_r2:.4f}</strong></div>
                            <div class="training-stat"><span>Learning rate</span><strong>{best_params['learning_rate']:.3f}</strong></div>
                            <div class="training-stat"><span>Tree depth</span><strong>{best_params['max_depth']}</strong></div>
                            <div class="training-stat"><span>Estimators</span><strong>{best_params['n_estimators']}</strong></div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if status_placeholder is not None:
                status_placeholder.markdown(
                    f"""
                    <div class="run-status-row">
                        <span>{scenario_index + 1:02d} / {scenario_total:02d}</span>
                        <strong>{safe_scenario_name}</strong>
                        <em>Optimization {i + 1:02d} / {n_iterations:02d}</em>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            if progress_placeholder is not None:
                progress_placeholder.progress(
                    (scenario_index + ((i + 1) / n_iterations)) / scenario_total
                )
            await asyncio.sleep(0.15 if show_optimization_ui else _UI_REFRESH_SECONDS)
        training_placeholder.empty()

        if not best_params:
            return (
                "Model optimization could not produce finite validation scores. "
                "Check that the historical target contains enough varying numeric values."
            )

        # Finally, train the best model on the FULL historical dataset
        model = train(X_hist, y_hist, params=best_params)
        r2 = best_cv_r2
        rmse = best_cv_rmse
        
    except Exception as e:
        return f"Model optimization failed: {e}"
        
    train_time = time.time() - t0

    # Advanced mode keeps its detailed live optimization feedback. Basic mode
    # stays focused on the compact scenario-level progress panel.
    finalize_msg = st.empty() if show_optimization_ui else _NoOp()
    if show_optimization_ui:
        finalize_msg.markdown(
            """
            <div class="validation-strip">
                <div class="validation-strip__index">FINAL</div>
                <div>
                    <strong>Finalizing predictions</strong>
                    <p>Calculating scenario output and SHAP explanations.</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    await asyncio.sleep(_UI_REFRESH_SECONDS)

    # 6. Prediction: Predict the missing target variable in the scenario
    preds = predict(model, X_scen)
    scen_filled = scen_df.copy()
    scen_filled[target_col] = preds

    # 8. Explainability: Calculate SHAP values to identify key drivers
    try:
        mean_shap, feat_names = compute_shap_values(model, X_scen)
    except Exception as e:
        logging.error(f"SHAP computation failed for {scenario_name}: {e}")
        mean_shap, feat_names = None, None

    # Clear the finalizing message once done
    finalize_msg.empty()

    return dict(
        target_col=target_col,
        hist_df=hist_df,
        scen_filled=scen_filled,
        r2=r2,
        rmse=rmse,
        train_time=train_time,
        n_features=len(X_hist.columns),
        mean_shap=mean_shap,
        feat_names=feat_names,
        model=model,
        seasonal=seasonal,
        search_iterations=search_iterations,
        cv_folds=cv_folds,
        depth_range=depth_range,
        rolling_windows=rolling_windows,
    )


# ---------------------------------------------------------------------------
# Main content area
# ---------------------------------------------------------------------------
if st.session_state.get("show_doc", False):
    render_documentation(EXAMPLE_FILE_BYTES)
    st.stop()

if uploaded is None:
    # Empty state - Interactive Hero via components.html
    components.html(
        """
        <!DOCTYPE html>
        <html>
        <head>
        <meta charset="utf-8">
        <style>
          @import url('https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400..700;1,6..72,400..700&family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
          
          :root {
            --paper: #F9F8F3;
            --ink: #191919;
            --grid-line: rgba(230, 226, 216, 0.40);
            --blue: #1648d8;
            --mono: 'Space Grotesk', monospace;
            --sans: 'Inter', system-ui, -apple-system, sans-serif;
            --serif: 'Newsreader', Georgia, serif;
          }
          
          * { box-sizing: border-box; margin: 0; padding: 0; }
          body {
            background: var(--paper);
            color: var(--ink);
            font-family: var(--sans);
            overflow: hidden;
            user-select: none;
            -webkit-user-select: none;
          }

          .hero-shell {
            position: relative;
            width: 100%;
            height: 350px;
            padding: 2.2rem 2.5rem;
            border: 1px solid #c9c5ba;
            background:
              linear-gradient(var(--grid-line) 1px, transparent 1px),
              linear-gradient(90deg, var(--grid-line) 1px, transparent 1px),
              var(--paper);
            background-size: 28px 28px;
            background-position: -1px -1px;
            overflow: hidden;
          }

          .hero-kicker {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            margin-bottom: 0.9rem;
            color: var(--ink);
            font-family: var(--mono);
            font-size: 11px;
            letter-spacing: 0.08em;
            text-transform: uppercase;
          }
          .hero-kicker::before {
            content: "";
            width: 24px;
            height: 1px;
            background: var(--ink);
          }

          h1 {
            max-width: 680px;
            margin: 0;
            color: var(--ink);
            font-family: var(--serif);
            font-size: clamp(38px, 5.5vw, 68px);
            font-weight: 400;
            line-height: 0.92;
            letter-spacing: -0.055em;
          }
          h1 span span { color: var(--blue); }

          p.hero-desc {
            max-width: 580px;
            margin: 1.1rem 0 0;
            color: #525049;
            font-size: 15px;
            line-height: 1.55;
          }

          .hero-flow {
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 0.5rem;
            margin-top: 1.6rem;
          }
          .hero-flow span {
            padding: 0.45rem 0.7rem;
            border: 1px solid var(--ink);
            background: var(--paper);
            font-family: var(--mono);
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 0.03em;
          }
          .hero-flow b {
            color: var(--blue);
            font-size: 12px;
          }

          .signal {
            width: 150px;
            height: 150px;
            aspect-ratio: 1;
            padding: 0;
            border: 0;
            background: var(--blue);
            border-radius: 50%;
            position: absolute;
            left: 0;
            top: 0;
            z-index: 10;
            cursor: grab;
            touch-action: none;
            mix-blend-mode: multiply;
            outline: none;
            will-change: transform;
          }
          .signal:active, .signal.is-dragging { cursor: grabbing; }
          .signal::before, .signal::after {
            content: "";
            position: absolute;
            inset: 22%;
            border: 1px solid rgba(255, 255, 255, 0.85);
            border-radius: 50%;
            pointer-events: none;
          }
          .signal::after {
            inset: 43%;
            background: var(--paper);
            border: 0;
          }
        </style>
        </head>
        <body>
          <div class="hero-shell">
            <div class="hero-kicker">Universal forecasting &amp; scenario intelligence</div>
            <h1>Turn historical data into <span>clear <span style="color:#1648d8;">scenarios.</span></span></h1>
            <p class="hero-desc">
              Upload a structured Excel workbook and the forecasting engine will validate the data,
              train a time-aware model, project every scenario, and explain what drove the result across any domain.
            </p>
            <div class="hero-flow">
              <span>01 · Upload workbook</span><b>→</b>
              <span>02 · Train & validate</span><b>→</b>
              <span>03 · Compare outcomes</span>
            </div>
            <button class="signal" type="button" aria-label="Interactive blue ball" title="Drag me or hover to hit me down"></button>
          </div>

          <script>
            const ball = document.querySelector(".signal");
            const court = document.querySelector(".hero-shell");
            if (ball && court) {
              const state = {
                x: 0, y: 0, vx: 0, vy: 0,
                dragging: false, sleeping: true,
                lastTime: 0, lastMoveTime: 0, lastMoveX: 0, lastMoveY: 0,
                offsetX: 0, offsetY: 0
              };
              const gravity = 2200;
              const bounce = 0.72;
              const wallBounce = 0.78;
              const padding = 18;
              const pointer = { x: 0, y: 0, previousX: 0, previousY: 0, lastTime: 0, ready: false, inside: false };
              let frame = 0;

              const bounds = () => ({
                minX: padding,
                maxX: Math.max(padding, court.clientWidth - ball.offsetWidth - padding),
                minY: padding,
                maxY: Math.max(padding, court.clientHeight - ball.offsetHeight - padding)
              });

              const clamp = (v, min, max) => Math.min(max, Math.max(min, v));
              const draw = () => { ball.style.transform = `translate3d(${state.x}px,${state.y}px,0)`; };

              const wake = () => {
                state.sleeping = false;
                state.lastTime = performance.now();
                cancelAnimationFrame(frame);
                frame = requestAnimationFrame(step);
              };

              const reset = () => {
                const limit = bounds();
                state.x = limit.maxX - 20;
                state.y = limit.minY + 20;
                state.vx = 0;
                state.vy = 0;
                state.sleeping = true;
                draw();
              };

              const step = time => {
                if (state.dragging || state.sleeping) return;
                const dt = Math.min((time - state.lastTime) / 1000, 0.025);
                state.lastTime = time;
                const limit = bounds();

                state.vy += gravity * dt;
                state.vx *= Math.pow(0.997, dt * 60);
                state.x += state.vx * dt;
                state.y += state.vy * dt;

                if (state.x <= limit.minX) {
                  state.x = limit.minX;
                  state.vx = Math.abs(state.vx) * wallBounce;
                } else if (state.x >= limit.maxX) {
                  state.x = limit.maxX;
                  state.vx = -Math.abs(state.vx) * wallBounce;
                }
                if (state.y <= limit.minY) {
                  state.y = limit.minY;
                  state.vy = Math.abs(state.vy) * wallBounce;
                } else if (state.y >= limit.maxY) {
                  state.y = limit.maxY;
                  state.vy = -Math.abs(state.vy) * bounce;
                  state.vx *= 0.92;
                  if (Math.abs(state.vy) < 65 && Math.abs(state.vx) < 18) {
                    state.vy = 0;
                    state.vx = 0;
                    state.sleeping = true;
                  }
                }
                draw();
                if (!state.sleeping) frame = requestAnimationFrame(step);
              };

              ball.addEventListener("pointerdown", event => {
                event.preventDefault();
                ball.setPointerCapture(event.pointerId);
                const courtRect = court.getBoundingClientRect();
                state.dragging = true;
                state.sleeping = true;
                state.vx = 0;
                state.vy = 0;
                state.offsetX = event.clientX - courtRect.left - state.x;
                state.offsetY = event.clientY - courtRect.top - state.y;
                state.lastMoveX = state.x;
                state.lastMoveY = state.y;
                state.lastMoveTime = performance.now();
                ball.classList.add("is-dragging");
              });

              ball.addEventListener("pointermove", event => {
                if (!state.dragging) return;
                const courtRect = court.getBoundingClientRect();
                const limit = bounds();
                const now = performance.now();
                const nextX = clamp(event.clientX - courtRect.left - state.offsetX, limit.minX, limit.maxX);
                const nextY = clamp(event.clientY - courtRect.top - state.offsetY, limit.minY, limit.maxY);
                const dt = Math.max((now - state.lastMoveTime) / 1000, 0.008);
                state.vx = state.vx * 0.35 + ((nextX - state.lastMoveX) / dt) * 0.65;
                state.vy = state.vy * 0.35 + ((nextY - state.lastMoveY) / dt) * 0.65;
                state.x = nextX;
                state.y = nextY;
                state.lastMoveX = nextX;
                state.lastMoveY = nextY;
                state.lastMoveTime = now;
                draw();
              });

              const release = event => {
                if (!state.dragging) return;
                state.dragging = false;
                ball.classList.remove("is-dragging");
                if (ball.hasPointerCapture(event.pointerId)) ball.releasePointerCapture(event.pointerId);
                state.vx = clamp(state.vx, -1700, 1700);
                state.vy = clamp(state.vy, -1900, 1900);
                wake();
              };

              ball.addEventListener("pointerup", release);
              ball.addEventListener("pointercancel", release);

              court.addEventListener("pointermove", event => {
                if (state.dragging || event.buttons !== 0) return;
                const courtRect = court.getBoundingClientRect();
                const now = performance.now();
                const x = event.clientX - courtRect.left;
                const y = event.clientY - courtRect.top;
                if (!pointer.ready) {
                  pointer.x = x; pointer.y = y; pointer.previousX = x; pointer.previousY = y; pointer.lastTime = now; pointer.ready = true;
                  return;
                }

                const moveX = x - pointer.x;
                const moveY = y - pointer.y;
                const moveLengthSquared = moveX * moveX + moveY * moveY;
                const radius = ball.offsetWidth / 2;
                const centerX = state.x + radius;
                const centerY = state.y + radius;
                const segmentPosition = moveLengthSquared
                  ? clamp(((centerX - pointer.x) * moveX + (centerY - pointer.y) * moveY) / moveLengthSquared, 0, 1)
                  : 0;
                const closestX = pointer.x + moveX * segmentPosition;
                const closestY = pointer.y + moveY * segmentPosition;
                const distance = Math.hypot(centerX - closestX, centerY - closestY);
                const currentDistance = Math.hypot(centerX - x, centerY - y);
                const colliding = distance <= radius + 8;
                const dt = Math.max((now - pointer.lastTime) / 1000, 0.008);
                const cursorVX = moveX / dt;
                const cursorVY = moveY / dt;
                const cursorSpeed = Math.hypot(cursorVX, cursorVY);

                if (colliding && !pointer.inside && cursorSpeed > 60) {
                  const directionX = cursorVX / cursorSpeed;
                  const directionY = cursorVY / cursorSpeed;
                  const impact = Math.min(cursorSpeed, 1650);
                  state.vx = state.vx * 0.2 + directionX * impact * 0.92;
                  state.vy = state.vy * 0.2 + directionY * impact * 0.92;
                  const limit = bounds();
                  state.x = clamp(state.x + directionX * Math.min(16, impact * 0.012), limit.minX, limit.maxX);
                  state.y = clamp(state.y + directionY * Math.min(16, impact * 0.012), limit.minY, limit.maxY);
                  draw();
                  wake();
                  pointer.inside = true;
                } else if (currentDistance > radius + 20) {
                  pointer.inside = false;
                }

                pointer.previousX = pointer.x;
                pointer.previousY = pointer.y;
                pointer.x = x;
                pointer.y = y;
                pointer.lastTime = now;
              });

              court.addEventListener("pointerleave", () => {
                pointer.ready = false; pointer.inside = false;
              });

              window.addEventListener("resize", () => {
                const limit = bounds();
                state.x = clamp(state.x, limit.minX, limit.maxX);
                state.y = clamp(state.y, limit.minY, limit.maxY);
                draw();
              });

              setTimeout(reset, 50);
            }
          </script>
        </body>
        </html>
        """,
        height=365,
        scrolling=False,
    )

    st.markdown(
        """
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-top: 24px;">
            <div class="ui-card">
                <div style="margin-bottom: 12px;">
                    <span class="ui-badge">Time-Aware Validation</span>
                </div>
                <h3 style="font-family: 'Newsreader', Georgia, serif; font-size: 22px; font-weight: 600; margin: 0 0 8px 0; color: var(--text-primary);">TimeSeriesSplit CV</h3>
                <p style="font-family: 'Inter', sans-serif; font-size: 15px; color: var(--text-secondary); line-height: 1.5; margin: 0;">
                    Uses forward-only expanding window cross-validation to strictly prevent future observations from leaking into past training iterations.
                </p>
            </div>
            <div class="ui-card">
                <div style="margin-bottom: 12px;">
                    <span class="ui-badge">Scenario Engine</span>
                </div>
                <h3 style="font-family: 'Newsreader', Georgia, serif; font-size: 22px; font-weight: 600; margin: 0 0 8px 0; color: var(--text-primary);">Multi-Sheet Intelligence</h3>
                <p style="font-family: 'Inter', sans-serif; font-size: 15px; color: var(--text-secondary); line-height: 1.5; margin: 0;">
                    Processes historical observations and projects multiple future scenario sheets simultaneously from a single Excel workbook.
                </p>
            </div>
            <div class="ui-card">
                <div style="margin-bottom: 12px;">
                    <span class="ui-badge">Game-Theoretic XAI</span>
                </div>
                <h3 style="font-family: 'Newsreader', Georgia, serif; font-size: 22px; font-weight: 600; margin: 0 0 8px 0; color: var(--text-primary);">TreeSHAP Attribution</h3>
                <p style="font-family: 'Inter', sans-serif; font-size: 15px; color: var(--text-secondary); line-height: 1.5; margin: 0;">
                    Surfaces exact driver feature importances for every step, showing what factors moved the needle on future projections.
                </p>
            </div>
        </div>

        <div class="ui-card" style="margin-top: 24px;">
            <div style="max-width: 880px; margin-bottom: 24px;">
                <span class="ui-badge">Why this platform exists</span>
                <h4 style="font-family: 'Newsreader', Georgia, serif; font-size: 32px; font-weight: 600; line-height: 1.1; margin: 14px 0 10px; color: var(--text-primary);">The forecasting problem we solve</h4>
                <p style="font-family: 'Inter', sans-serif; font-size: 16px; color: var(--text-secondary); line-height: 1.65; margin: 0;">
                    Organizations collect years of measurements, but turning that history into a trustworthy answer to
                    <strong>“what is likely to happen under this future scenario?”</strong> is difficult. The data must be cleaned,
                    time order must be protected, useful memory and seasonal signals must be created, models must be compared,
                    and every result must be explained. A technically valid forecast is not enough—it also has to be understandable
                    and usable by the people making the decision.
                </p>
            </div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px;">
                <div style="padding: 20px; background: var(--bg-card-alt); border: 1px solid var(--border-light);">
                    <p style="font-family: 'JetBrains Mono', monospace; font-size: 11px; letter-spacing: .08em; text-transform: uppercase; color: var(--primary); margin: 0 0 10px;">The traditional way</p>
                    <h5 style="font-family: 'Inter', sans-serif; font-size: 18px; margin: 0 0 10px; color: var(--text-primary);">Many manual steps, often across several tools</h5>
                    <p style="font-family: 'Inter', sans-serif; font-size: 14px; line-height: 1.6; color: var(--text-secondary); margin: 0;">
                        Analysts commonly prepare data in spreadsheets, choose and code a statistical or machine-learning model,
                        design time-aware validation, tune parameters, build charts, and then create a separate explanation for stakeholders.
                        Each handoff adds time and creates opportunities for inconsistent assumptions or accidental future-data leakage.
                    </p>
                </div>
                <div style="padding: 20px; background: var(--bg-card-alt); border: 1px solid var(--border-light);">
                    <p style="font-family: 'JetBrains Mono', monospace; font-size: 11px; letter-spacing: .08em; text-transform: uppercase; color: var(--primary); margin: 0 0 10px;">MATLAB and other platforms</p>
                    <h5 style="font-family: 'Inter', sans-serif; font-size: 18px; margin: 0 0 10px; color: var(--text-primary);">Powerful, but specialist setup is often required</h5>
                    <p style="font-family: 'Inter', sans-serif; font-size: 14px; line-height: 1.6; color: var(--text-secondary); margin: 0;">
                        MATLAB, Python notebooks, R, and dedicated forecasting packages can produce excellent results. However, teams may
                        need programming expertise, appropriate licenses or toolboxes, model-specific configuration, and custom work to connect
                        cleaning, scenario forecasting, validation, explainability, visualization, and export into one repeatable workflow.
                    </p>
                </div>
                <div style="padding: 20px; background: var(--primary-light); border: 1px solid var(--primary-border);">
                    <p style="font-family: 'JetBrains Mono', monospace; font-size: 11px; letter-spacing: .08em; text-transform: uppercase; color: var(--primary); margin: 0 0 10px;">Our approach</p>
                    <h5 style="font-family: 'Inter', sans-serif; font-size: 18px; margin: 0 0 10px; color: var(--text-primary);">One guided workflow from Excel to an explained scenario</h5>
                    <p style="font-family: 'Inter', sans-serif; font-size: 14px; line-height: 1.6; color: var(--text-secondary); margin: 0;">
                        Upload one structured workbook. The platform validates the sheets, identifies the forecast target, reviews data quality,
                        builds time and memory features, evaluates XGBoost with forward-only cross-validation, predicts every scenario, and shows
                        which factors influenced the result. The same process can be repeated without rebuilding an analysis from scratch.
                    </p>
                </div>
            </div>
            <div style="margin-top: 18px; padding: 16px 18px; border-left: 3px solid var(--primary); background: var(--bg-card);">
                <p style="font-family: 'Inter', sans-serif; font-size: 14px; line-height: 1.6; color: var(--text-secondary); margin: 0;">
                    <strong style="color: var(--text-primary);">Where this is better:</strong> speed, accessibility, consistency, explainability,
                    and scenario comparison for structured tabular time-series data. Specialist tools such as MATLAB remain the better choice when
                    a project requires custom physical models, Simulink integration, advanced signal processing, or a fully bespoke research method.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# ---------------------------------------------------------------------------
# Process uploaded file
# ---------------------------------------------------------------------------
try:
    sheets = load_excel(uploaded)
    historical_raw, scenarios = separate_sheets(sheets)
except ValueError as e:
    st.markdown(
        f"""
        <div style="padding:1rem 1.5rem;background:var(--error-container);border-left:4px solid var(--error);border-radius:var(--rounded);margin-bottom:1.5rem;">
            <h3 style="margin:0 0 0.25rem 0;color:var(--error);font-size:16px;">File Structure Error</h3>
            <p class="body-sm" style="margin:0;color:var(--on-error-container);">{e}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()
except Exception as e:
    st.markdown(
        f"""
        <div style="padding:1rem 1.5rem;background:var(--error-container);border-left:4px solid var(--error);border-radius:var(--rounded);margin-bottom:1.5rem;">
            <h3 style="margin:0 0 0.25rem 0;color:var(--error);font-size:16px;">File Reading Error</h3>
            <p class="body-sm" style="margin:0;color:var(--on-error-container);">Failed to read the Excel file: {e}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

if not scenarios:
    st.markdown(
        """
        <div style="padding:1rem 1.5rem;background:var(--surface-container-highest);border-left:4px solid var(--outline);border-radius:var(--rounded);margin-bottom:1.5rem;">
            <h3 style="margin:0 0 0.25rem 0;color:var(--primary);font-size:16px;">No Scenarios Found</h3>
            <p class="body-sm" style="margin:0;color:var(--on-surface-variant);">Your Excel file must have at least 2 sheets: Sheet 1 (Historical) + Sheet 2+ (Scenarios)</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

if app_mode == "Advanced":
    st.markdown(
        f"""
        <div class="workflow-summary">
            <div class="workflow-summary__label">Workbook ready</div>
            <div class="workflow-summary__item">
                <span>Sheets loaded</span>
                <strong>{len(sheets)}</strong>
            </div>
            <div class="workflow-summary__item">
                <span>Historical sets</span>
                <strong>1</strong>
            </div>
            <div class="workflow-summary__item">
                <span>Scenarios found</span>
                <strong>{len(scenarios)}</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Data Quality Check
# ---------------------------------------------------------------------------
if app_mode == "Advanced":
    st.markdown(
        """
        <div class="section-heading">
            <div class="section-heading__number">01</div>
            <div>
                <h2>Data quality</h2>
                <p>Review placeholder rules and statistical flags before the model starts training.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# --- Settings: Advanced shows full controls, Basic uses safe defaults ---
# Defaults used in Basic mode (the Advanced controls below override these when shown).
iqr_threshold = 3.0
show_outlier_details = False
missing_placeholders = [-999.0, -9999.0, 999.0, 9999.0]
model_search_iterations = 15
model_cv_folds = 4
model_depth_range = (2, 8)
model_rolling_windows = (3, 7)

if app_mode == "Advanced":
    # Allow users to customize missing data placeholders
    st.markdown(
        """
        <div class="section-heading">
            <div class="section-heading__number">01.A</div>
            <div>
                <h2>Missing-value rules</h2>
                <p>Define sentinel values that should be converted to empty measurements.</p>
            </div>
        </div>
        <div class="validation-strip">
            <div class="validation-strip__index">NOTE</div>
            <div>
                <strong>Negative values may be valid measurements</strong>
                <p>-1 is not treated as missing by default. Add it only when your data specification explicitly uses it as a placeholder.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        default_missing = st.checkbox(
            "Use default missing values (-999, -9999, 999, 9999)",
            value=True,
            help="Common placeholders used to indicate missing data in scientific datasets"
        )

    with col2:
        custom_missing = st.text_input(
            "Additional missing values (comma-separated)",
            placeholder="e.g., -1, 0, 99999",
            help="Add any other values that should be treated as missing data"
        )

    # Outlier detection settings
    st.markdown(
        """
        <div class="section-heading">
            <div class="section-heading__number">01.B</div>
            <div>
                <h2>Outlier sensitivity</h2>
                <p>Adjust how aggressively the review flags measurements outside the expected statistical range.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col3, col4 = st.columns(2)

    with col3:
        iqr_threshold = st.slider(
            "IQR Threshold",
            min_value=1.5,
            max_value=5.0,
            value=3.0,
            step=0.5,
            help="Higher values = fewer outliers detected. 3.0 is standard, 1.5 is very strict, 5.0 is very lenient."
        )

    with col4:
        show_outlier_details = st.checkbox(
            "Show detailed outlier information",
            value=True,
            help="Display sample values and ranges for detected outliers"
        )

    st.markdown(
        """
        <div class="section-heading">
            <div class="section-heading__number">02</div>
            <div>
                <h2>Expert model tuning</h2>
                <p>Optional quality controls for experienced users. The recommended defaults are designed to balance accuracy, stability, and runtime.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Open expert tuning · Recommended defaults active", expanded=False):
        st.markdown(
            """
            **How to use this area**

            Change one setting at a time and compare the cross-validated R² and
            RMSE. A higher training score alone does not guarantee a better
            future forecast; stable validation performance is the goal.

            **Recommended workflow:** begin with Balanced settings. Try
            Conservative complexity when validation is unstable, add longer
            context windows only when the measured system reacts slowly, and
            use Thorough search for the final run after the other choices are
            settled.
            """
        )

        tune_col1, tune_col2 = st.columns(2)
        with tune_col1:
            search_effort = st.select_slider(
                "Search effort",
                options=["Quick", "Balanced", "Thorough"],
                value="Balanced",
                help=(
                    "Controls how many XGBoost configurations are tested. "
                    "Thorough can find a better model but takes roughly twice as long."
                ),
            )
            model_search_iterations = {
                "Quick": 8,
                "Balanced": 15,
                "Thorough": 30,
            }[search_effort]

            validation_rigor = st.select_slider(
                "Time-series validation",
                options=["3 folds", "4 folds", "5 folds"],
                value="4 folds",
                help=(
                    "More folds test the model across more historical cut-off dates. "
                    "Use 5 folds for long datasets; use 3 when history is limited."
                ),
            )
            model_cv_folds = int(validation_rigor.split()[0])

        with tune_col2:
            model_complexity = st.select_slider(
                "Model complexity",
                options=["Conservative", "Balanced", "Flexible"],
                value="Balanced",
                help=(
                    "Controls the tree-depth search range. Conservative models "
                    "reduce overfitting; Flexible models can capture more complex "
                    "relationships but need more data."
                ),
            )
            model_depth_range = {
                "Conservative": (2, 5),
                "Balanced": (2, 8),
                "Flexible": (2, 10),
            }[model_complexity]

            selected_windows = st.multiselect(
                "Rolling context windows",
                options=[3, 7, 14, 30],
                default=[3, 7],
                help=(
                    "Adds smoothed predictor values over this many observations. "
                    "Short windows follow quick changes; 14 or 30 observations "
                    "can help when the system responds more slowly."
                ),
            )
            model_rolling_windows = tuple(selected_windows)

        estimated_fits = model_search_iterations * model_cv_folds
        context_text = (
            ", ".join(str(window) for window in model_rolling_windows)
            if model_rolling_windows
            else "None"
        )
        st.markdown(
            f"""
            <div class="tuning-readout">
                <div><span>Candidate models</span><strong>{model_search_iterations}</strong></div>
                <div><span>Validation fits</span><strong>{estimated_fits}</strong></div>
                <div><span>Context windows</span><strong>{context_text}</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Build the missing placeholders list (Advanced only — Basic keeps the fixed defaults above)
    missing_placeholders = []
    if default_missing:
        missing_placeholders = [-999.0, -9999.0, 999.0, 9999.0]

    if custom_missing:
        try:
            custom_values = [float(x.strip()) for x in custom_missing.split(',') if x.strip()]
            missing_placeholders.extend(custom_values)
        except ValueError:
            st.warning("Invalid format in custom missing values. Please use comma-separated numbers.")

all_sheets = {"Historical": historical_raw, **scenarios}
issues_report = detect_data_issues(all_sheets, missing_placeholders=missing_placeholders if missing_placeholders else None, iqr_threshold=iqr_threshold)

total_issues = sum(sheet['total_issues'] for sheet in issues_report.values())
basic_validation_notice = None

if app_mode == "Advanced" and total_issues > 0:
    st.markdown(
        f"""
        <div class="issue-banner">
            <div class="issue-banner__count">{total_issues}<small>flags found</small></div>
            <div class="issue-banner__copy">
                <strong>Measurements need review</strong>
                <p>These are statistical flags, not automatic errors. Keep valid observations or select only the columns that should be cleaned.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Track which columns to clean
    columns_to_clean = {
        'missing_placeholders': [],
        'outliers': []
    }

    # Display detailed issues with per-column controls
    for sheet_name, sheet_issues in issues_report.items():
        if sheet_issues['total_issues'] == 0:
            continue

        raw_sheet_name = sheet_name.removeprefix("Scenario: ")
        sheet_display = _display_field_name(raw_sheet_name)
        if sheet_name.startswith("Scenario: "):
            sheet_display = f"Scenario · {sheet_display}"

        with st.expander(f"{sheet_display} — {sheet_issues['total_issues']} flags", expanded=False):
            # Missing placeholders
            if sheet_issues['missing_placeholders']:
                st.markdown("### Missing data placeholders")
                st.markdown("*Values that represent missing data:*")
                for col, placeholders in sheet_issues['missing_placeholders'].items():
                    column_display = _display_field_name(col)
                    with st.container():
                        col_row1, col_row2 = st.columns([3, 1])
                        with col_row1:
                            st.markdown(f"**{column_display}**")
                            for p in placeholders:
                                st.markdown(f"- `{p['value']}` appears **{p['count']}** times ({p['percentage']:.1f}%)")
                        with col_row2:
                            clean_this = st.checkbox(
                                f"Clean {column_display}",
                                key=f"clean_missing_{sheet_name}_{col}",
                                value=True,
                                help="Replace these values with NaN"
                            )
                            if clean_this and col not in columns_to_clean['missing_placeholders']:
                                columns_to_clean['missing_placeholders'].append((sheet_name, col))

            # Outliers
            if sheet_issues['outliers']:
                st.markdown("### Statistical outliers")
                st.markdown(f"*Values outside the normal range (IQR threshold: {iqr_threshold}):*")
                for col, outlier_info in sheet_issues['outliers'].items():
                    column_display = _display_field_name(col)
                    with st.container():
                        col_row1, col_row2 = st.columns([3, 1])
                        with col_row1:
                            st.markdown(f"**{column_display}**: **{outlier_info['count']}** outliers ({outlier_info['percentage']:.1f}%)")
                            st.markdown(f"  - Normal range: `{outlier_info['lower_bound']:.2f}` to `{outlier_info['upper_bound']:.2f}`")
                            if show_outlier_details and outlier_info['outlier_values']:
                                st.markdown(f"  - Sample outliers: `{', '.join(f'{v:.2f}' for v in outlier_info['outlier_values'][:5])}`")
                        with col_row2:
                            clean_this = st.checkbox(
                                f"Clean {column_display}",
                                key=f"clean_outlier_{sheet_name}_{col}",
                                value=False,  # Default to False for outliers since they might be valid
                                help="Replace these values with NaN"
                            )
                            if clean_this and col not in columns_to_clean['outliers']:
                                columns_to_clean['outliers'].append((sheet_name, col))

            # Extreme values
            if sheet_issues['extreme_values']:
                st.markdown("### Extreme value ranges")
                st.markdown("*Columns with unusually large value ranges:*")
                for col, ext_info in sheet_issues['extreme_values'].items():
                    column_display = _display_field_name(col)
                    st.markdown(f"- **{column_display}**: Range `{ext_info['min']:.2f}` to `{ext_info['max']:.2f}`")
                    st.markdown(f"  - Mean: `{ext_info['mean']:.2f}`, Std: `{ext_info['std']:.2f}`")

            # Recommendations
            if sheet_issues['recommendations']:
                st.markdown("### Recommendations")
                for rec in sheet_issues['recommendations']:
                    st.markdown(f"- {rec}")

    # Summary of what will be cleaned
    st.markdown(
        """
        <div class="section-heading">
            <div class="section-heading__number">01.C</div>
            <div>
                <h2>Review decision</h2>
                <p>Confirm which flagged columns should be cleaned before model execution.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    total_to_clean = len(columns_to_clean['missing_placeholders']) + len(columns_to_clean['outliers'])

    if total_to_clean > 0:
        st.markdown(f"**You've selected {total_to_clean} column(s) to clean:**")

        if columns_to_clean['missing_placeholders']:
            st.markdown("**Missing Data Placeholders:**")
            for sheet_name, col in columns_to_clean['missing_placeholders']:
                st.markdown(f"- {sheet_name}: {col}")

        if columns_to_clean['outliers']:
            st.markdown("**Statistical Outliers:**")
            for sheet_name, col in columns_to_clean['outliers']:
                st.markdown(f"- {sheet_name}: {col}")
    else:
        st.markdown(
            """
            <div class="selection-summary">
                <strong>No cleaning selected</strong>
                <p>All measurements will be sent to the model exactly as provided.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Action buttons
    col1, col2 = st.columns(2)

    with col1:
        clean_data = st.button(
            f"Clean selected ({total_to_clean})",
            type="primary",
            width="stretch",
            disabled=total_to_clean == 0,
            help="Replace selected problematic values with NaN"
        )

    with col2:
        proceed_as_is = st.button(
            "Proceed as provided",
            width="stretch",
            help="Continue with original data"
        )

    if clean_data:
        st.markdown(
            f"""
            <div class="validation-strip">
                <div class="validation-strip__index">READY</div>
                <div>
                    <strong>Data cleaning applied</strong>
                    <p>{total_to_clean} selected column(s) were cleaned. Model execution can begin.</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Clean specific columns based on user selection
        cleaned_sheets = clean_specific_columns(
            all_sheets,
            issues_report,
            columns_to_clean,
            missing_placeholders=missing_placeholders if missing_placeholders else None
        )

        # Update the main data structures
        historical_raw = cleaned_sheets["Historical"]
        scenarios = {name: cleaned_sheets[name] for name in scenarios.keys()}

    elif not proceed_as_is:
        st.warning("Choose either “Clean selected” or “Proceed as provided” to continue.")
        st.stop()

else:
    if app_mode == "Basic":
        # Basic mode: auto-clean only default sentinel placeholders across all
        # sheets, leave valid outliers untouched, and proceed without a gate.
        basic_validation_notice = st.empty()
        if total_issues > 0:
            # Replace default sentinel placeholders with NaN across all sheets.
            # (Outliers are left intact — they may be valid measurements.)
            all_sheets = {name: df.replace(missing_placeholders, np.nan) for name, df in all_sheets.items()}
            historical_raw = all_sheets["Historical"]
            scenarios = {name: all_sheets[name] for name in scenarios.keys()}
            placeholder_count = sum(
                placeholder["count"]
                for sheet_issues in issues_report.values()
                for placeholders in sheet_issues["missing_placeholders"].values()
                for placeholder in placeholders
            )
            outlier_count = sum(
                outlier["count"]
                for sheet_issues in issues_report.values()
                for outlier in sheet_issues["outliers"].values()
            )
            basic_validation_notice.markdown(
                f"""
                <div class="validation-strip">
                    <div class="validation-strip__index">CHECK</div>
                    <div>
                        <strong>Dataset validated</strong>
                        <p>{placeholder_count} missing placeholder(s) corrected. {outlier_count} statistical flag(s) retained as valid measurements.</p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            basic_validation_notice.markdown(
                """
                <div class="validation-strip">
                    <div class="validation-strip__index">CHECK</div>
                    <div>
                        <strong>Dataset validated</strong>
                        <p>No missing placeholders or statistical flags were detected.</p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            """
            <div class="validation-strip">
                <div class="validation-strip__index">CLEAR</div>
                <div>
                    <strong>No statistical flags found</strong>
                    <p>The workbook passed the configured data-quality rules and is ready for model execution.</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# Automatically run ALL scenarios
# ---------------------------------------------------------------------------
async def _run_scenarios_and_render() -> None:
    """Train and validate every scenario, then render one result tab each."""
    results: dict[str, dict] = {}
    errors: dict[str, str] = {}

    scenario_items = list(scenarios.items())
    scenario_total = len(scenario_items)
    run_header = st.empty()
    run_status = st.empty()
    run_header.markdown(
        f"""
        <div class="run-stage">
            <div class="run-stage__kicker">Model execution</div>
            <div class="run-stage__copy">
                <h2>Running time-series scenarios</h2>
                <p>{scenario_total} scenario{"s" if scenario_total != 1 else ""} queued for time-series training and validation.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    progress_bar = st.progress(0)

    for idx, (scen_name, scen_df) in enumerate(scenario_items):
        scenario_label = html.escape(scen_name.removeprefix("Scenario: "))
        progress_bar.progress(idx / scenario_total)
        run_status.markdown(
            f"""
            <div class="run-status-row">
                <span>{idx + 1:02d} / {scenario_total:02d}</span>
                <strong>{scenario_label}</strong>
                <em>Training &amp; validating model</em>
            </div>
            """,
            unsafe_allow_html=True,
        )
        result = await _run_scenario(
            historical_raw.copy(),
            scen_df.copy(),
            scen_name,
            seasonal=seasonal_mode,
            show_optimization_ui=app_mode == "Advanced",
            status_placeholder=run_status,
            progress_placeholder=progress_bar,
            scenario_index=idx,
            scenario_total=scenario_total,
            search_iterations=model_search_iterations,
            cv_folds=model_cv_folds,
            depth_range=model_depth_range,
            rolling_windows=model_rolling_windows,
        )
        if isinstance(result, str):
            errors[scen_name] = result
        else:
            results[scen_name] = result
            # Log successful simulation results for future AI analysis
            save_simulation_log(scen_name, result)
        progress_bar.progress((idx + 1) / scenario_total)

    run_header.empty()
    run_status.empty()
    progress_bar.empty()
    if basic_validation_notice is not None:
        basic_validation_notice.empty()

    # Show errors if any
    for scen_name, err_msg in errors.items():
        st.error(f"**{scen_name}** — {err_msg}")

    if not results:
        st.error("All scenarios failed. Please check your Excel file.")
        st.stop()

    # -----------------------------------------------------------------------
    # Render results — one tab per scenario (auto-generated)
    # -----------------------------------------------------------------------
    st.markdown(
        f"""
        <div class="result-overview">
            <div>
                <div class="hero-kicker">Simulation complete</div>
                <h1>Scenario results</h1>
            </div>
            <p>{len(results)} scenario(s) processed with time-series cross-validation and explainable feature attribution.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    tab_names = list(results.keys())
    tabs = st.tabs(
        [_display_field_name(name.removeprefix("Scenario: ")) for name in tab_names]
    )

    for tab, scen_name in zip(tabs, tab_names):
        res = results[scen_name]
        scenario_display = _display_field_name(scen_name.removeprefix("Scenario: "))
        target_display = _display_field_name(res["target_col"])
        scenario_display_html = html.escape(scenario_display)
        target_display_html = html.escape(target_display)

        with tab:
            # ── Header ─────────────────────────────────────────────────
            col_title, col_download = st.columns([4, 1])
            with col_title:
                st.markdown(
                    f"""
                    <div style="margin-bottom:0.3rem;">
                        <span class="status-dot live"></span>
                        <span class="label-caps" style="color:var(--primary);">Prediction Complete</span>
                    </div>
                    <h2 style="margin:0;">
                        {scenario_display_html}
                    </h2>
                    """,
                    unsafe_allow_html=True,
                )
            with col_download:
                # Use German formatting: semicolon separator and comma for decimals
                csv_bytes = res["scen_filled"].to_csv(sep=';', decimal=',').encode("utf-8")
                safe_name = scen_name.replace("Scenario: ", "").replace(" ", "_")
                st.download_button(
                    label="Download CSV",
                    data=csv_bytes,
                    file_name=f"{safe_name}_predicted.csv",
                    mime="text/csv",
                    key=f"dl_{scen_name}",
                )

            # ── Metrics strip ──────────────────────────────────────────
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy (R²)", f"{res['r2']:.4f}", help="R-squared score. Measures how well the model predicts the target. 1.0 is perfect, 0.0 means it's just guessing the average.")
            m2.metric("Avg Error (RMSE)", f"{res['rmse']:.5f}", help="Root Mean Squared Error. The average absolute difference between the predicted and actual values. Lower is better.")
            m3.metric("Target Variable", target_display, help=f"Source column: {res['target_col']}")
            m4.metric("Factors", str(res["n_features"]), help="The total number of historical data columns (including engineered features like lags) the model used to make its prediction.")

            st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

            # ── Main chart: time-series comparison ─────────────────────
            st.markdown(
                f'''
                <div class="eco-card">
                    <h3 title="This chart compares the historical data (light blue) against the AI's projections for the target variable in the scenario (orange dashed line).">
                        Target Variable Prediction 
                        <span class="material-symbols-outlined" style="font-size:16px; vertical-align:middle; color:var(--outline); cursor:help;">info</span>
                    </h3>
                ''',
                unsafe_allow_html=True,
            )
            fig_ts = plot_comparison(res["hist_df"], res["scen_filled"], res["target_col"])
            st.plotly_chart(fig_ts, width="stretch", config={"displayModeBar": True})
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

            # ── Bottom section: SHAP + Data Summary side by side ───────
            if app_mode == "Advanced":
                col_shap, col_summary = st.columns([2, 1], gap="medium")
            else:
                # Basic: SHAP graph full-width; expert scorecard & metadata are hidden.
                col_shap = st.container()

            with col_shap:
                st.markdown(
                    f'''
                    <div class="eco-card">
                        <h3 title="SHAP (SHapley Additive exPlanations) values show which historical factors most heavily influenced the AI's predictions.">
                            AI Decision Factors (SHAP)
                            <span class="material-symbols-outlined" style="font-size:16px; vertical-align:middle; color:var(--outline); cursor:help;">info</span>
                        </h3>
                        <p class="body-sm" style="color:var(--on-surface-variant); margin-bottom:1rem;">Which factors moved the needle? Longer bars mean a larger impact.</p>
                    ''',
                    unsafe_allow_html=True,
                )
                if res["mean_shap"] is not None:
                    fig_shap = plot_shap_bar(res["mean_shap"], res["feat_names"])
                    st.plotly_chart(fig_shap, width="stretch", config={"displayModeBar": False})
                else:
                    st.info("AI Logic Breakdown is not available for this scenario.")
                st.markdown("</div>", unsafe_allow_html=True)

            if app_mode == "Advanced":
                with col_summary:
                    st.markdown(
                        f"""
                        <div class="score-stack">
                            <div class="score-stack__header">Model scorecard</div>
                            <div class="score-stack__item" title="Root Mean Squared Error. Lower is better.">
                                <span>Average error / RMSE</span>
                                <strong>{res['rmse']:.5f}</strong>
                            </div>
                            <div class="score-stack__item" title="R-squared score. 1.0 is a perfect fit.">
                                <span>Validation R²</span>
                                <strong>{res['r2']:.4f}</strong>
                            </div>
                            <div class="score-stack__item score-stack__item--target" title="{html.escape(res['target_col'])}">
                                <span>Predicted variable</span>
                                <strong>{target_display_html}</strong>
                            </div>
                            <div class="score-stack__item">
                                <span>Training time</span>
                                <strong>{res['train_time']:.1f}s</strong>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            if app_mode == "Advanced":
                # ── Footer: model metadata ─────────────────────────────────
                st.markdown(
                    f"""
                    <div class="model-facts">
                        <div class="model-fact" title="The machine-learning algorithm used for this forecast.">
                            <span>Model</span>
                            <strong>XGBoost regressor</strong>
                        </div>
                        <div class="model-fact" title="Training and validation completed without errors.">
                            <span>Training status</span>
                            <strong class="model-status">Converged</strong>
                        </div>
                        <div class="model-fact" title="Total model optimization and training time.">
                            <span>Calculation time</span>
                            <strong>{res['train_time']:.1f} seconds</strong>
                        </div>
                        <div class="model-fact" title="Historical records used to train the model.">
                            <span>Training volume</span>
                            <strong>{len(res['hist_df']):,} rows</strong>
                        </div>
                        <div class="model-fact" title="Forward-only splits used to estimate performance on unseen future periods.">
                            <span>Validation design</span>
                            <strong>{res['cv_folds']} forward folds</strong>
                        </div>
                        <div class="model-fact" title="Number of candidate XGBoost configurations evaluated.">
                            <span>Search effort</span>
                            <strong>{res['search_iterations']} candidates</strong>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# ---------------------------------------------------------------------------
# A Streamlit server runs the coroutine to completion here. In the browser,
# Pyodide's event loop is already running and cannot be blocked, so the
# stlite entrypoint (web/entrypoint.py) awaits `scenario_run` instead.
# ---------------------------------------------------------------------------
scenario_run = _run_scenarios_and_render()
if not IS_BROWSER:
    asyncio.run(scenario_run)
