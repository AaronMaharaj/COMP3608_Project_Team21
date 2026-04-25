import os
from typing import Tuple

import numpy as np
import pandas as pd


def load_alzheimers(
    filepath: str = "data/raw/oasis_longitudinal.csv",
) -> Tuple[pd.DataFrame, pd.Series]:
    """Load and clean the OASIS Alzheimer's longitudinal dataset.

    Uses only baseline visits (Visit == 1) and enforces strict binary
    classification by excluding 'Converted' patients.

    Args:
        filepath: Path to the raw CSV file.

    Returns:
        Tuple of (features, target) where target is 1=Demented, 0=Nondemented.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}.")

    df = pd.read_csv(filepath)

    # Use only baseline visits to prevent the same patient appearing in train and test splits
    df = df.loc[df["Visit"] == 1].copy()

    # Enforce strict binary classification by excluding 'Converted' patients.
    # These patients progressed over time and create label ambiguity that confuses the model.
    df = df[df["Group"].isin(["Demented", "Nondemented"])].copy()
    df["Group"] = df["Group"].map({"Demented": 1, "Nondemented": 0})
    df["M/F"] = df["M/F"].map({"M": 1, "F": 0})

    # Remove identifiers and columns that cause target leakage (CDR is directly used to assign Group)
    cols_to_drop = ["Subject ID", "MRI ID", "Hand", "CDR", "Visit", "MR Delay"]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    df = df.dropna(subset=["Group"])

    X = df.drop("Group", axis=1)
    y = df["Group"].astype(int)

    return X, y


def load_parkinsons_v2(
    filepath: str = "data/raw/pd_speech_features.csv",
) -> Tuple[pd.DataFrame, pd.Series]:
    """Load Parkinson's dataset with one recording per patient (deduplicated).

    .. deprecated::
        Use :func:`load_parkinsons_v3` instead. This loader discards biological
        variance across recordings and is only useful for quick single-recording
        experiments. The main pipeline uses v3 with GroupKFold.

    Args:
        filepath: Path to the raw CSV file.

    Returns:
        Tuple of (features, target).
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}.")

    df = pd.read_csv(filepath)

    # Sort by 'id' to guarantee we keep the baseline recordings deterministically,
    # then keep only the first recording to ensure no patient appears in both train and test.
    df = df.sort_values("id").drop_duplicates(subset=["id"], keep="first").copy()

    # Drop rows with missing or corrupted audio parsing values.
    df = df.dropna()
    df = df.drop(columns=["id"])

    # Split features and target.
    X = df.drop("class", axis=1)
    y = df["class"]

    return X, y


def load_parkinsons_v3(
    filepath: str = "data/raw/pd_speech_features.csv",
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Load the full Sakar Parkinson's dataset with all recordings per patient.

    Unlike v2 which deduplicates to 1 recording per patient, this loader
    retains all 3 recordings to preserve biological variance (fatigue, stress).
    Returns the patient id as a group vector for use with GroupKFold/GroupShuffleSplit
    to prevent data leakage.

    Args:
        filepath: Path to the raw CSV file.

    Returns:
        Tuple of (features, target, groups) where groups is the patient id Series.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}.")

    df = pd.read_csv(filepath)
    df = df.dropna()

    groups = df["id"]
    X = df.drop(columns=["class", "id"])
    y = df["class"]

    return X, y, groups


def load_autism(
    filepath: str = "data/raw/autism_screening.csv",
) -> Tuple[pd.DataFrame, pd.Series]:
    """Load and clean the Autism Screening dataset.

    Target Variable: 'Class/ASD' (1 = YES, 0 = NO).

    Args:
        filepath: Path to the raw CSV file.

    Returns:
        Tuple of (features, target).
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}.")

    df = pd.read_csv(filepath)

    # Strip whitespace from column names.
    df.columns = df.columns.str.strip()

    # Map target variable: yes/no to 1/0.
    if "Class/ASD" in df.columns:
        df["Class/ASD"] = (
            df["Class/ASD"].astype(str).str.upper().map({"YES": 1, "NO": 0})
        )

    # Drop 'result' because it is literally the sum of A1-A10 (direct target leakage).
    # Drop "age_desc" because it's a constant string in this dataset ('18 and more') and provides zero variance.
    # Fix: bring back a1-a10 since they are the primary behavioral signals
    cols_to_drop = ["result", "age_desc"]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # In case the dataset uses '?' for missing values, replace with NaN.
    df = df.replace("?", np.nan)

    # Do not drop NaN rows here; missingness is handled downstream in the pipeline.

    # Isolate features (X) and target (y).
    X = df.drop("Class/ASD", axis=1)
    y = df["Class/ASD"]

    return X, y
