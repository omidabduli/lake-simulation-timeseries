"""The scenario pipeline in app.py must be reproducible.

app.py cannot be imported in tests (it renders the Streamlit page at import
time), so `_run_scenario` is extracted from its source and run with a stub
in place of the Streamlit placeholders.
"""
from __future__ import annotations

import ast
import asyncio
import html
import logging
import random
import sys
import time
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from modules import data_loader, feature_engineering, model
from modules.explainer import compute_shap_values

ROOT = Path(__file__).resolve().parent.parent


class _Absorb:
    def __getattr__(self, _name):
        return lambda *args, **kwargs: None


class _StreamlitStub:
    def empty(self):
        return _Absorb()


def _load_run_scenario(train):
    wanted = {"_run_scenario", "_NoOp", "IS_BROWSER", "_UI_REFRESH_SECONDS"}
    tree = ast.parse((ROOT / "app.py").read_text(encoding="utf-8"))
    nodes = [
        node
        for node in tree.body
        if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name in wanted)
        or (isinstance(node, ast.Assign) and any(getattr(t, "id", None) in wanted for t in node.targets))
    ]
    namespace = {
        "st": _StreamlitStub(), "np": np, "pd": pd, "html": html, "time": time, "random": random,
        "logging": logging, "asyncio": asyncio, "sys": sys, "TimeSeriesSplit": TimeSeriesSplit,
        "r2_score": r2_score, "mean_squared_error": mean_squared_error,
        "validate_column_match": data_loader.validate_column_match,
        "detect_target_column": data_loader.detect_target_column,
        "normalize_time_index": data_loader.normalize_time_index,
        "engineer_features": feature_engineering.engineer_features,
        "train": train, "predict": model.predict, "compute_shap_values": compute_shap_values,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "app.py", "exec"), namespace)
    return namespace["_run_scenario"]


def _workbook():
    rng = np.random.default_rng(3)
    dates = pd.date_range("2024-01-01", periods=150, freq="D")
    temperature = 12 + 8 * np.sin(np.arange(150) / 20) + rng.normal(0, 1, 150)
    historical = pd.DataFrame({
        "Date": dates,
        "Temperature": temperature,
        "Oxygen": 11 - 0.25 * temperature + rng.normal(0, 0.2, 150),
    })
    scenario = pd.DataFrame({
        "Date": pd.date_range("2024-05-30", periods=30, freq="D"),
        "Temperature": temperature[-30:] + 1.5,
        "Oxygen": np.nan,
    })
    return historical, scenario


class HyperparameterSearchTests(unittest.TestCase):
    def _run(self, interfere: bool):
        """Return the candidate parameters evaluated, in order, and the final scores."""
        candidates = []

        def train(X, y, params=None):
            if interfere:
                random.getrandbits(32)  # what a websockets keep-alive ping does
            candidates.append(dict(params or {}))
            return model.train(X, y, params=params)

        historical, scenario = _workbook()
        result = asyncio.run(_load_run_scenario(train)(
            historical, scenario, "Scenario: test",
            show_optimization_ui=False, search_iterations=4, cv_folds=2,
        ))
        self.assertIsInstance(result, dict, result)
        return candidates, result["r2"], result["rmse"]

    def test_search_ignores_other_users_of_the_global_random_state(self) -> None:
        """Web-server threads (e.g. WebSocket keep-alive pings) draw from `random`."""
        undisturbed = self._run(interfere=False)
        disturbed = self._run(interfere=True)
        self.assertEqual(len(undisturbed[0]), 4 * 2 + 1)  # 4 candidates x 2 folds + final fit
        self.assertEqual(disturbed, undisturbed)


if __name__ == "__main__":
    unittest.main()
