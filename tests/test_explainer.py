from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
import pandas as pd

from modules import explainer
from modules.model import train


def _fitted_model_and_frame():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            "temperature": rng.normal(15, 5, 240),
            "wind": rng.normal(4, 1, 240),
            "radiation": rng.normal(200, 60, 240),
        }
    )
    frame.loc[::17, "wind"] = np.nan  # XGBoost's native missing-value handling
    target = 0.3 * frame["temperature"] - 0.8 * frame["wind"].fillna(4) + rng.normal(0, 0.5, 240)
    model = train(frame, target, params={"n_estimators": 60, "max_depth": 4})
    return model, frame


class ExplainerTests(unittest.TestCase):
    def test_xgboost_tree_shap_matches_shap_tree_explainer(self) -> None:
        """The browser fallback must reproduce shap.TreeExplainer exactly."""
        if explainer.shap is None:
            self.skipTest("shap is not installed")
        model, frame = _fitted_model_and_frame()

        expected = explainer.shap.TreeExplainer(model).shap_values(frame)
        actual = explainer.xgboost_tree_shap_values(model, frame)

        np.testing.assert_array_equal(actual, expected)

    def test_compute_shap_values_without_shap_package(self) -> None:
        model, frame = _fitted_model_and_frame()
        with mock.patch.object(explainer, "shap", None):
            mean_abs_shap, feature_names = explainer.compute_shap_values(model, frame)

        self.assertEqual(feature_names, list(frame.columns))
        contributions = explainer.xgboost_tree_shap_values(model, frame)
        np.testing.assert_array_equal(mean_abs_shap, np.abs(contributions).mean(axis=0))

    def test_tree_shap_values_are_additive(self) -> None:
        """Local accuracy: contributions plus the bias term equal the raw prediction."""
        model, frame = _fitted_model_and_frame()
        import xgboost as xgb

        full = model.get_booster().predict(xgb.DMatrix(frame), pred_contribs=True)
        np.testing.assert_allclose(full.sum(axis=1), model.predict(frame), rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
