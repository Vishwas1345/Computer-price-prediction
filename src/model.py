"""Price prediction model wrapper.

Wraps a point-estimate regressor plus two quantile regressors so every
prediction comes with a confidence range, not just a bare number.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge

from .features import SpecFeatureEncoder, engineer_features

RANDOM_STATE = 42

# Interval bounds: a 90% central prediction interval.
LOWER_QUANTILE, UPPER_QUANTILE = 0.05, 0.95


def make_regressor(name: str):
    """Factory for the algorithms compared in training."""
    if name == "linear":
        return LinearRegression()
    if name == "ridge":
        return Ridge(alpha=1.0, random_state=RANDOM_STATE)
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=100, n_jobs=-1, random_state=RANDOM_STATE
        )
    if name == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(
            max_iter=500, learning_rate=0.08, random_state=RANDOM_STATE
        )
    raise ValueError(f"Unknown algorithm: {name}")


class PricePredictor:
    """Point-estimate model plus quantile models for confidence ranges."""

    def __init__(self, algorithm: str = "hist_gradient_boosting"):
        self.algorithm = algorithm
        self.model = make_regressor(algorithm)
        self.lower_model = HistGradientBoostingRegressor(
            loss="quantile", quantile=LOWER_QUANTILE,
            max_iter=300, random_state=RANDOM_STATE,
        )
        self.upper_model = HistGradientBoostingRegressor(
            loss="quantile", quantile=UPPER_QUANTILE,
            max_iter=300, random_state=RANDOM_STATE,
        )

    def train(self, X: pd.DataFrame, y: pd.Series) -> "PricePredictor":
        self.model.fit(X, y)
        self.lower_model.fit(X, y)
        self.upper_model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # Prices are positive; clip protects against extrapolation artifacts.
        return np.clip(self.model.predict(X), 0, None)

    def predict_with_confidence(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict price with a 90% confidence range per row."""
        point = self.predict(X)
        lower = np.clip(self.lower_model.predict(X), 0, None)
        upper = self.upper_model.predict(X)
        # Quantile models are trained independently, so enforce ordering.
        lower = np.minimum(lower, point)
        upper = np.maximum(upper, point)
        return pd.DataFrame(
            {"price": point, "lower": lower, "upper": upper}, index=X.index
        )

    def feature_importance(
        self, X: pd.DataFrame, y: pd.Series, max_samples: int = 2000
    ) -> pd.Series:
        """Permutation importance (algorithm-agnostic), on a sample for speed."""
        if len(X) > max_samples:
            X = X.sample(max_samples, random_state=RANDOM_STATE)
            y = y.loc[X.index]
        result = permutation_importance(
            self.model, X, y, n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1
        )
        return pd.Series(
            result.importances_mean, index=X.columns
        ).sort_values(ascending=False)

    def explain_prediction(
        self, encoder: SpecFeatureEncoder, user_specs: dict
    ) -> pd.DataFrame:
        """Attribute the prediction to each spec the user chose.

        For each spec, swap the user's value into a typical configuration
        (the training-set median/mode for that device type) and record how
        the predicted price moves. Positive delta = this spec pushes the
        price up relative to a typical machine.

        Correlated fields are swapped as a group: the GPU's tier and VRAM
        move with the GPU model, and the CPU's tier/cores/clocks move with
        the CPU family — otherwise the model (which prices mostly off the
        tier numerics) would attribute nothing to the named component.
        """
        groups = {
            "gpu_model": ["gpu_model", "gpu_brand", "gpu_tier", "vram_gb"],
            "cpu_family": [
                "cpu_family", "cpu_brand", "cpu_tier", "cpu_cores",
                "cpu_threads", "cpu_base_ghz", "cpu_boost_ghz",
            ],
        }
        grouped_fields = {f for fields in groups.values() for f in fields}

        device = user_specs.get("device_type", "Laptop")
        baseline = dict(encoder.defaults_by_device_[device])
        baseline["device_type"] = device
        baseline_price = self._predict_raw(encoder, baseline)

        units = [(lead, fields) for lead, fields in groups.items()] + [
            (spec, [spec])
            for spec in baseline
            if spec != "device_type" and spec not in grouped_fields
        ]

        rows = []
        for lead, fields in units:
            present = [f for f in fields if f in user_specs]
            if not present or all(user_specs[f] == baseline[f] for f in present):
                continue
            variant = dict(baseline)
            variant.update({f: user_specs[f] for f in present})
            delta = self._predict_raw(encoder, variant) - baseline_price
            rows.append({"spec": lead, "value": user_specs.get(lead), "delta": delta})

        return (
            pd.DataFrame(rows, columns=["spec", "value", "delta"])
            .sort_values("delta", ascending=False)
            .reset_index(drop=True)
        )

    def _predict_raw(self, encoder: SpecFeatureEncoder, specs: dict) -> float:
        """Predict from a raw spec dict (pre-encoding)."""
        row = pd.DataFrame([_with_derivable_columns(specs)])
        return float(self.predict(encoder.transform(row))[0])

    def save(self, path) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path) -> "PricePredictor":
        return joblib.load(path)


def _with_derivable_columns(specs: dict) -> dict:
    """Fill derived columns if the caller supplied only the raw ones.

    App input supplies `cpu_family` and `megapixels` directly (no
    `model`/`cpu_model`/`resolution` columns); dataset rows supply the raw
    columns instead. Either form must encode identically.
    """
    from .features import extract_cpu_family, resolution_to_megapixels

    specs = dict(specs)
    if "cpu_family" not in specs:
        specs["cpu_family"] = extract_cpu_family(specs.get("cpu_model", ""))
    if "megapixels" not in specs:
        specs["megapixels"] = resolution_to_megapixels(specs.get("resolution", ""))
    return specs


def predict_from_specs(
    predictor: PricePredictor, encoder: SpecFeatureEncoder, specs: dict
) -> dict:
    """One-stop inference for app/tests: raw spec dict -> price + range."""
    row = pd.DataFrame([_with_derivable_columns(specs)])
    encoded = encoder.transform(row)
    result = predictor.predict_with_confidence(encoded).iloc[0]
    return {
        "price": float(result["price"]),
        "lower": float(result["lower"]),
        "upper": float(result["upper"]),
    }


def load_artifacts(models_dir: str | Path):
    """Load the trained model and fitted encoder saved by train.py."""
    models_dir = Path(models_dir)
    predictor = PricePredictor.load(models_dir / "model.pkl")
    encoder = SpecFeatureEncoder.load(models_dir / "encoders.pkl")
    return predictor, encoder
