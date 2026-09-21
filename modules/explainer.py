"""
explainer.py — SHAP-based model explanation using TreeExplainer.

The `shap` package is used when it is installed (local / server runs). The
in-browser build (Pyodide) cannot install it because it depends on numba, so
there the same TreeSHAP values come from XGBoost's built-in implementation —
the exact call shap.TreeExplainer itself makes for XGBoost models.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

try:
    import shap
except ImportError:
    shap = None


def compute_shap_values(
    model: xgb.XGBRegressor,
    X_scenario: pd.DataFrame,
) -> tuple[np.ndarray, list[str]]:
    """Compute mean absolute SHAP values for each feature.

    SHAP (SHapley Additive exPlanations) values assign an importance score
    to each feature for each prediction. It is based on cooperative game theory,
    fairly allocating the 'payout' (the prediction deviation from baseline) 
    among the player features.

    Uses shap.TreeExplainer, a high-performance variant optimized specifically 
    for tree-based ensembles (like XGBoost, LightGBM, and Random Forests).
    Unlike kernel explainer approximations, TreeExplainer computes exact SHAP
    values in polynomial time, making it exceptionally fast even with many features.

    Parameters
    ----------
    model : xgb.XGBRegressor
        The fitted XGBoost model.
    X_scenario : pd.DataFrame
        The scenario feature matrix (same columns used during training).

    Returns
    -------
    (mean_abs_shap, feature_names)
        mean_abs_shap : np.ndarray shape (n_features,) — average absolute |SHAP| impact per feature.
        feature_names : list[str] — corresponding feature names.
    """
    if shap is not None:
        # Initialize the TreeExplainer with the trained XGBoost model
        explainer = shap.TreeExplainer(model)

        # Calculate SHAP values for the scenario data (yields matrix of shape [n_samples, n_features])
        shap_values = explainer.shap_values(X_scenario)      # shape (n_samples, n_features)
    else:
        shap_values = xgboost_tree_shap_values(model, X_scenario)

    # Compute the average absolute impact of each feature across all scenario timesteps
    mean_abs_shap = np.abs(shap_values).mean(axis=0)         # shape (n_features,)
    feature_names = list(X_scenario.columns)

    return mean_abs_shap, feature_names


def xgboost_tree_shap_values(
    model: xgb.XGBRegressor,
    X_scenario: pd.DataFrame,
) -> np.ndarray:
    """Exact TreeSHAP values computed by XGBoost itself.

    Mirrors shap.TreeExplainer(model).shap_values(X) for XGBoost models
    (tree_path_dependent, no background data): all trees, raw margin output.
    XGBoost appends the bias term as the last column, which is dropped.
    """
    contributions = model.get_booster().predict(
        xgb.DMatrix(X_scenario),
        iteration_range=(0, 0),
        pred_contribs=True,
        validate_features=False,
    )
    return contributions[:, :-1]
