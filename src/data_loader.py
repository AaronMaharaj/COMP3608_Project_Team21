import pandas as pd
import os
import numpy as np


def load_alzheimers(filepath="data/raw/oasis_longitudinal.csv"):
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

    print(f"Loaded OASIS Alzheimer's. Shape: {X.shape}")
    return X, y


def load_parkinsons_v2(filepath="data/raw/pd_speech_features.csv"):
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

    print(f"Loaded Parkinson's (Sakar). Shape: {X.shape}")
    return X, y


def load_autism(filepath="data/raw/autism_screening.csv"):
    """
    Loads and cleans the Autism Screening dataset.
    Target Variable: 'Class/ASD' (1 = YES, 0 = NO)
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
    print(
        f"   [Autism] {df.isnull().any(axis=1).sum()} rows contain missing values — handled by pipeline imputer"
    )

    # Isolate features (X) and target (y).
    X = df.drop("Class/ASD", axis=1)
    y = df["Class/ASD"]

    print(f"[Data Loader] Autism Screening Dataset Loaded. Shape: {X.shape}")
    return X, y
