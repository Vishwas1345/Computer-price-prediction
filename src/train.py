"""Train and evaluate price prediction models.

Usage:
    python -m src.train [--data PATH] [--quick]

Compares several regression algorithms on a held-out test set, trains the
best one (with quantile models for confidence ranges), saves artifacts to
models/, and writes evaluation charts and metrics to results/.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from .data_loader import DEFAULT_DATA_PATH, ROOT, load_data, split_features_target
from .features import SpecFeatureEncoder
from .model import RANDOM_STATE, PricePredictor, make_regressor

MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

ALGORITHMS = ["linear", "ridge", "random_forest", "hist_gradient_boosting"]


def compare_algorithms(X_train, y_train, X_test, y_test) -> pd.DataFrame:
    rows = []
    for name in ALGORITHMS:
        regressor = make_regressor(name)
        start = time.time()
        regressor.fit(X_train, y_train)
        pred = regressor.predict(X_test)
        rows.append(
            {
                "algorithm": name,
                "r2_test": r2_score(y_test, pred),
                "mae_test": mean_absolute_error(y_test, pred),
                "r2_train": regressor.score(X_train, y_train),
                "fit_seconds": round(time.time() - start, 1),
            }
        )
        print(f"  {rows[-1]}")
    return pd.DataFrame(rows).sort_values("r2_test", ascending=False)


def save_charts(predictor, encoder, X_test, y_test, raw_test, importance):
    RESULTS_DIR.mkdir(exist_ok=True)
    pred = predictor.predict(X_test)
    residuals = y_test - pred

    fig, ax = plt.subplots(figsize=(8, 6))
    top = importance.head(20).iloc[::-1]
    ax.barh(top.index, top.values, color="steelblue")
    ax.set_title("Permutation feature importance (top 20)")
    ax.set_xlabel("Mean decrease in R² when shuffled")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "feature_importance.png", dpi=120)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_test, pred, s=4, alpha=0.25)
    lims = [min(y_test.min(), pred.min()), max(y_test.max(), pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1)
    ax.set_xlabel("Actual price ($)")
    ax.set_ylabel("Predicted price ($)")
    ax.set_title("Actual vs predicted (test set)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "actual_vs_predicted.png", dpi=120)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(pred, residuals, s=4, alpha=0.25)
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Predicted price ($)")
    ax.set_ylabel("Residual (actual − predicted, $)")
    ax.set_title("Residuals vs predicted (test set)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "residuals.png", dpi=120)

    fig, ax = plt.subplots(figsize=(8, 5))
    mae_by_brand = (
        residuals.abs().groupby(raw_test["brand"]).mean().sort_values()
    )
    ax.barh(mae_by_brand.index, mae_by_brand.values, color="darkseagreen")
    ax.set_xlabel("Mean absolute error ($)")
    ax.set_title("Prediction error by brand (test set)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "error_by_brand.png", dpi=120)

    plt.close("all")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=DEFAULT_DATA_PATH, help="dataset path")
    parser.add_argument(
        "--quick", action="store_true",
        help="skip the algorithm comparison; train the final model only",
    )
    args = parser.parse_args()

    print("Loading data...")
    df = load_data(args.data)
    X_raw, y = split_features_target(df)
    raw_train, raw_test, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # Fit the encoder on the training split only, so no test-set categories
    # or statistics leak into training.
    encoder = SpecFeatureEncoder().fit(raw_train)
    X_train = encoder.transform(raw_train)
    X_test = encoder.transform(raw_test)
    print(f"Encoded {X_train.shape[1]} features from {raw_train.shape[1]} columns.")

    comparison = None
    if not args.quick:
        print("Comparing algorithms...")
        comparison = compare_algorithms(X_train, y_train, X_test, y_test)
        print(comparison.to_string(index=False))

    best = comparison.iloc[0]["algorithm"] if comparison is not None \
        else "hist_gradient_boosting"
    print(f"Training final model ({best}) with quantile models...")
    predictor = PricePredictor(algorithm=best).train(X_train, y_train)

    intervals = predictor.predict_with_confidence(X_test)
    covered = ((y_test >= intervals["lower"]) & (y_test <= intervals["upper"])).mean()
    metrics = {
        "algorithm": best,
        "r2_test": r2_score(y_test, intervals["price"]),
        "mae_test": mean_absolute_error(y_test, intervals["price"]),
        "interval_coverage_90pct_nominal": float(covered),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features_encoded": X_train.shape[1],
    }
    print(json.dumps(metrics, indent=2))

    print("Computing feature importance and saving charts...")
    importance = predictor.feature_importance(X_test, y_test)
    save_charts(predictor, encoder, X_test, y_test, raw_test, importance)

    MODELS_DIR.mkdir(exist_ok=True)
    predictor.save(MODELS_DIR / "model.pkl")
    encoder.save(MODELS_DIR / "encoders.pkl")

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    if comparison is not None:
        comparison.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    importance.head(30).to_csv(RESULTS_DIR / "feature_importance.csv")

    print(f"Saved model + encoder to {MODELS_DIR}/, results to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
