"""
visualizer.py — Plotly chart builders for the Scenario Simulator.

Design system:
Colors: warm paper (#F9F8F3), card (#FFFFFF), primary accent (#1648d8).
Font: Inter for UI, Space Grotesk / JetBrains Mono for data labels.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------
_SURFACE = "#F9F8F3"
_CARD_BG = "#FFFFFF"
_BORDER = "#E6E2D8"
_GRID = "#EAE6DD"
_TEXT_PRIMARY = "#191919"
_TEXT_SECONDARY = "#555555"
_TEXT_MUTED = "#707070"

_HIST_COLOR = "#191919"      # Deep charcoal — historical actual
_PRED_COLOR = "#1648d8"      # Primary blue accent — scenario predicted
_SECONDARY = "#1648d8"       # Primary blue accent — feature impact bars
_ERROR = "#ef4444"           # Red status indicator


def _base_layout(**kwargs) -> dict:
    return dict(
        paper_bgcolor=_CARD_BG,
        plot_bgcolor=_CARD_BG,
        font=dict(
            color=_TEXT_SECONDARY,
            family="Inter, system-ui, -apple-system, sans-serif",
            size=14,
        ),
        margin=dict(l=58, r=24, t=46, b=46),
        xaxis=dict(
            gridcolor=_GRID,
            showline=True,
            linecolor=_BORDER,
            zeroline=False,
            tickfont=dict(family="Space Grotesk, JetBrains Mono, monospace", size=11, color=_TEXT_MUTED),
        ),
        yaxis=dict(
            gridcolor=_GRID,
            showline=True,
            linecolor=_BORDER,
            zeroline=False,
            tickfont=dict(family="Space Grotesk, JetBrains Mono, monospace", size=11, color=_TEXT_MUTED),
        ),
        legend=dict(
            bgcolor="rgba(255,255,255,0)",
            bordercolor=_BORDER,
            borderwidth=0,
            font=dict(family="Inter, sans-serif", size=12, color=_TEXT_SECONDARY),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        hoverlabel=dict(
            bgcolor=_CARD_BG,
            font_family="Space Grotesk, JetBrains Mono, monospace",
            font_size=13,
            bordercolor=_BORDER,
        ),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Chart 1 — Time-series comparison
# ---------------------------------------------------------------------------

def plot_comparison(
    hist_df: pd.DataFrame,
    scenario_df: pd.DataFrame,
    target_col: str,
) -> go.Figure:
    """Interactive line chart: historical actual vs. scenario predicted."""
    fig = go.Figure()

    # Historical — solid line
    fig.add_trace(go.Scatter(
        x=hist_df.index,
        y=hist_df[target_col],
        mode="lines",
        name="Historical Data",
        line=dict(color=_HIST_COLOR, width=2.5, shape="spline", smoothing=0.35),
        hovertemplate="%{x|%Y-%m-%d}<br><b>%{y:.4f}</b><extra>Historical</extra>",
    ))

    # Scenario predicted — dashed line
    fig.add_trace(go.Scatter(
        x=scenario_df.index,
        y=scenario_df[target_col],
        mode="lines",
        name="Scenario Projection",
        line=dict(color=_PRED_COLOR, width=2.8, dash="dash", shape="spline", smoothing=0.35),
        hovertemplate="%{x|%Y-%m-%d}<br><b>%{y:.4f}</b><extra>Predicted</extra>",
    ))

    layout = _base_layout(
        hovermode="x unified",
        height=380,
    )
    fig.update_layout(**layout)
    return fig


# ---------------------------------------------------------------------------
# Chart 2 — SHAP feature importance
# ---------------------------------------------------------------------------

def _readable_feature_name(name: str) -> str:
    """Turn engineered feature keys into compact chart labels."""
    label = str(name)
    label = label.replace("_rolling_", " · rolling ")
    label = label.replace("_lag_", " · lag ")
    label = label.replace("doy_cos", "Day of year · cosine")
    label = label.replace("doy_sin", "Day of year · sine")
    label = label.replace("dow_cos", "Day of week · cosine")
    label = label.replace("dow_sin", "Day of week · sine")
    label = label.replace("_", " ")
    label = label.replace(" ug L", " µg/L").replace(" mg L", " mg/L")
    label = label.replace(" Temperature C", " temperature °C")
    return label


def plot_shap_bar(
    mean_abs_shap: np.ndarray,
    feature_names: list[str],
) -> go.Figure:
    """Horizontal bar chart of mean absolute SHAP values per feature."""
    # Sort ascending for horizontal bars (top = most important)
    order = np.argsort(mean_abs_shap)
    sorted_names = [_readable_feature_name(feature_names[i]) for i in order]
    sorted_vals = mean_abs_shap[order]

    # Only show top 10 features
    if len(sorted_names) > 10:
        sorted_names = sorted_names[-10:]
        sorted_vals = sorted_vals[-10:]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=sorted_vals,
        y=sorted_names,
        orientation="h",
        marker=dict(
            color=_SECONDARY,
            opacity=0.9,
            line=dict(width=0),
        ),
        hovertemplate="<b>%{y}</b><br>Mean |SHAP|: %{x:.5f}<extra></extra>",
        name="Feature Impact",
    ))

    layout = _base_layout(
        height=320,
        bargap=0.35,
        showlegend=False,
    )
    fig.update_layout(**layout)
    fig.update_yaxes(
        tickfont=dict(family="Space Grotesk, JetBrains Mono, monospace", size=10, color=_TEXT_SECONDARY),
    )
    fig.update_xaxes(
        title_text="Mean |SHAP Value|",
        title_font=dict(size=10, color=_TEXT_MUTED),
    )
    return fig
