"""Load and clean the computer price dataset."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = ROOT / "Computer price prediction" / "computer_prices_all.xlsx"

TARGET = "price"


def load_data(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the raw dataset from Excel (or CSV) and apply basic cleaning.

    Cleaning steps:
    - drop exact duplicate rows
    - drop rows with a missing or non-positive price
    - strip whitespace from string columns
    """
    path = Path(path)
    if path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)

    df = df.drop_duplicates()

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    df = df.dropna(subset=[TARGET])
    df = df[df[TARGET] > 0]

    return df.reset_index(drop=True)


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a loaded dataframe into features X and target y."""
    return df.drop(columns=[TARGET]), df[TARGET]
