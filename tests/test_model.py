"""Basic sanity tests for the trained model and encoder artifacts.

Run:  pytest tests/  (requires models/ artifacts from `python -m src.train`)
"""

from pathlib import Path

import pandas as pd
import pytest

from src.features import extract_cpu_family, resolution_to_megapixels
from src.model import load_artifacts, predict_from_specs

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"

pytestmark = pytest.mark.skipif(
    not (MODELS_DIR / "model.pkl").exists(),
    reason="run `python -m src.train` first to create model artifacts",
)


@pytest.fixture(scope="module")
def artifacts():
    return load_artifacts(MODELS_DIR)


@pytest.fixture()
def gaming_laptop_specs(artifacts):
    _, encoder = artifacts
    specs = dict(encoder.defaults_by_device_["Laptop"])
    specs.update(
        device_type="Laptop",
        brand="MSI",
        cpu_family="Intel i7",
        gpu_model="RTX 40 80",
        ram_gb=32,
        storage_type="NVMe",
        storage_gb=1024,
    )
    # As the app does, fill the component internals (tier, VRAM, cores...)
    # with the typical values for the chosen CPU/GPU.
    specs.update(encoder.cpu_profiles_["Intel i7"])
    specs.update(encoder.gpu_profiles_["RTX 40 80"])
    return specs


def test_artifacts_load(artifacts):
    predictor, encoder = artifacts
    assert hasattr(predictor, "predict")
    assert encoder.options_["brand"], "encoder should expose brand choices"


def test_predict_returns_sane_price(artifacts, gaming_laptop_specs):
    predictor, encoder = artifacts
    result = predict_from_specs(predictor, encoder, gaming_laptop_specs)
    # A 32GB / RTX 40 80 gaming laptop should land in a plausible band.
    assert 500 < result["price"] < 10000
    assert 0 <= result["lower"] <= result["price"] <= result["upper"]


def test_unseen_category_does_not_crash(artifacts, gaming_laptop_specs):
    """A GPU/CPU/brand never seen in training must degrade gracefully."""
    predictor, encoder = artifacts
    specs = dict(gaming_laptop_specs)
    specs.update(
        brand="Framework",           # brand not in the dataset
        cpu_family="Intel Ultra 9",  # future CPU family
        gpu_model="RTX 60 90",       # future GPU
    )
    result = predict_from_specs(predictor, encoder, specs)
    assert result["price"] > 0


def test_unseen_categories_are_reported(artifacts):
    _, encoder = artifacts
    row = dict(encoder.defaults_by_device_["Desktop"])
    row.update(device_type="Desktop", brand="Framework", model="X", resolution="1920x1080")
    unseen = encoder.unseen_categories(pd.DataFrame([row]))
    assert unseen == {"brand": ["Framework"]}


def test_explanation_directions_make_sense(artifacts, gaming_laptop_specs):
    predictor, encoder = artifacts
    breakdown = predictor.explain_prediction(encoder, gaming_laptop_specs)
    deltas = dict(zip(breakdown["spec"], breakdown["delta"]))
    # A high-end GPU should push price up versus a typical laptop.
    assert deltas.get("gpu_model", 0) > 0


def test_feature_helpers():
    assert extract_cpu_family("Intel i5-11129") == "Intel i5"
    assert extract_cpu_family("AMD Ryzen 7 6230") == "AMD Ryzen 7"
    assert extract_cpu_family("Apple M2 Pro") == "Apple M2 Pro"
    assert resolution_to_megapixels("1920x1080") == pytest.approx(2.07, abs=0.01)
    assert resolution_to_megapixels("not a resolution") == 0.0
