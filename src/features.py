"""Feature engineering and categorical encoding.

`SpecFeatureEncoder` is a fit-once, reuse-later transformer: fit it on the
training split, save it with the model, and apply the exact same encoding to
new user input at inference time. Unseen categories (a GPU or CPU that was
not in the training data) encode to all-zero one-hot columns instead of
raising, so the model degrades gracefully rather than crashing.
"""

from __future__ import annotations

import re

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# `model` is a marketing name that is unique for ~99% of rows — it acts as a
# row ID, so the encoder drops it. `cpu_model` (27k unique values) is replaced
# by the derived low-cardinality `cpu_family`. `resolution` is replaced by a
# numeric megapixel count.
DROP_COLUMNS = ["model", "cpu_model", "resolution"]

CATEGORICAL_COLUMNS = [
    "device_type",
    "brand",
    "os",
    "form_factor",
    "cpu_brand",
    "cpu_family",
    "gpu_brand",
    "gpu_model",
    "storage_type",
    "display_type",
    "wifi",
]

# Numeric specs that describe the chosen GPU/CPU rather than the machine.
# The encoder records typical values per GPU model / CPU family so the app
# can auto-fill them and explanations can vary them together.
GPU_PROFILE_COLUMNS = ["gpu_tier", "vram_gb"]
CPU_PROFILE_COLUMNS = [
    "cpu_tier", "cpu_cores", "cpu_threads", "cpu_base_ghz", "cpu_boost_ghz",
]


def extract_cpu_family(cpu_model: str) -> str:
    """Reduce a CPU model string to its family.

    'Intel i5-11129' -> 'Intel i5', 'AMD Ryzen 7 6230' -> 'AMD Ryzen 7',
    'Apple M2 Pro' -> 'Apple M2 Pro' (Apple names carry no numeric suffix).
    """
    family = re.sub(r"[\s-]*\d{3,5}\w*$", "", str(cpu_model)).strip()
    return family if family else "Unknown"


def resolution_to_megapixels(resolution: str) -> float:
    """'2560x1440' -> 3.69 (megapixels). Unparseable values become 0."""
    match = re.match(r"(\d+)\s*x\s*(\d+)", str(resolution))
    if not match:
        return 0.0
    return int(match.group(1)) * int(match.group(2)) / 1e6


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive model-ready raw features from the loaded dataset columns.

    Derived columns are only computed when absent, so callers (like the app)
    may supply `cpu_family`/`megapixels` directly instead of the raw columns.
    """
    df = df.copy()
    if "cpu_family" not in df.columns:
        df["cpu_family"] = df["cpu_model"].map(extract_cpu_family)
    if "megapixels" not in df.columns:
        df["megapixels"] = df["resolution"].map(resolution_to_megapixels)
    return df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])


class SpecFeatureEncoder:
    """One-hot encodes categorical spec columns, passes numerics through.

    Also records the category choices and numeric ranges seen during
    training, which the app uses to populate dropdowns and input bounds.
    """

    def __init__(self):
        self._onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self._numeric_columns: list[str] = []
        self.options_: dict[str, list] = {}
        self.resolutions_: list[str] = []
        self.numeric_ranges_: dict[str, dict] = {}
        self.defaults_by_device_: dict[str, dict] = {}
        self.gpu_profiles_: dict[str, dict] = {}
        self.cpu_profiles_: dict[str, dict] = {}

    def fit(self, df: pd.DataFrame) -> "SpecFeatureEncoder":
        engineered = engineer_features(df)
        self._numeric_columns = [
            c for c in engineered.columns if c not in CATEGORICAL_COLUMNS
        ]
        self._onehot.fit(engineered[CATEGORICAL_COLUMNS])

        self.options_ = {
            col: sorted(engineered[col].unique().tolist())
            for col in CATEGORICAL_COLUMNS
        }
        if "resolution" in df.columns:
            self.resolutions_ = sorted(
                df["resolution"].unique(), key=resolution_to_megapixels
            )
        self.gpu_profiles_ = (
            engineered.groupby("gpu_model")[GPU_PROFILE_COLUMNS]
            .median().to_dict(orient="index")
        )
        self.cpu_profiles_ = (
            engineered.groupby("cpu_family")[CPU_PROFILE_COLUMNS]
            .median().to_dict(orient="index")
        )
        self.numeric_ranges_ = {
            col: {
                "min": float(engineered[col].min()),
                "max": float(engineered[col].max()),
                "median": float(engineered[col].median()),
            }
            for col in self._numeric_columns
        }
        # Typical configuration per device type, used by the app as form
        # defaults and by explanations as the comparison baseline.
        for device, group in engineered.groupby("device_type"):
            defaults = group.median(numeric_only=True).to_dict()
            for col in CATEGORICAL_COLUMNS:
                defaults[col] = group[col].mode().iloc[0]
            self.defaults_by_device_[device] = defaults
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        engineered = engineer_features(df)
        onehot = self._onehot.transform(engineered[CATEGORICAL_COLUMNS])
        onehot_names = self._onehot.get_feature_names_out(CATEGORICAL_COLUMNS)
        numeric = engineered[self._numeric_columns].astype(float)
        return pd.DataFrame(
            np.hstack([numeric.to_numpy(), onehot]),
            columns=self._numeric_columns + list(onehot_names),
            index=engineered.index,
        )

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    @property
    def feature_names_(self) -> list[str]:
        return self._numeric_columns + list(
            self._onehot.get_feature_names_out(CATEGORICAL_COLUMNS)
        )

    def unseen_categories(self, df: pd.DataFrame) -> dict[str, list]:
        """Report which values in `df` were never seen during training."""
        engineered = engineer_features(df)
        unseen = {}
        for col in CATEGORICAL_COLUMNS:
            novel = set(engineered[col].unique()) - set(self.options_[col])
            if novel:
                unseen[col] = sorted(novel)
        return unseen

    def save(self, path) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path) -> "SpecFeatureEncoder":
        return joblib.load(path)
