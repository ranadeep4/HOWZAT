"""Data loading utilities for IPL Score Prediction pipeline."""
from pathlib import Path
import pandas as pd


def load_ipl_data(path: str = None) -> pd.DataFrame:
    """Load IPL CSV file.

    Args:
        path: path to csv file. If None, uses `content/ipl_data.csv` relative to project root.

    Returns:
        DataFrame
    """
    if path is None:
        path = Path(__file__).resolve().parents[1] / "content" / "ipl_data.csv"
    else:
        path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"CSV file not found at {path}")

    df = pd.read_csv(path)
    return df
