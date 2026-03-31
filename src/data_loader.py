import pandas as pd
import os


def load_alzheimers(filepath='data/raw/oasis_longitudinal.csv'):
    """
    Loads and cleans the OASIS Alzheimer's longitudinal dataset.
    Target Variable: 'Group' (1 = Demented, 0 = Nondemented)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}.")

    df = pd.read_csv(filepath)

    # Keep only visit 1 so we don't leak patient data between train/test splits
    df = df.loc[df['Visit'] == 1].copy()

    # Drop 'Convert' patients so we just have a clean Binary Classification problem
    df = df[df['Group'].isin(['Demented', 'Nondemented'])].copy()
    df['Group'] = df['Group'].map({'Demented': 1, 'Nondemented': 0})

    # Map M/F to 1/0
    df['M/F'] = df['M/F'].map({'M': 1, 'F': 0})

    # Fill missing SES (socioeconomic status) with the median SES of patients who share the same Education level
    df['SES'] = df.groupby('EDUC')['SES'].transform(lambda x: x.fillna(x.median()))
    
    # If any NaNs remain (e.g., in MMSE), safely drop them
    df = df.dropna()

    # Drop identifiers and variables that might cause target leakage (like CDR)
    cols_to_drop = ['Subject ID', 'MRI ID', 'Hand', 'CDR', 'Visit']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Isolate Features (X) and Target (y)
    X = df.drop('Group', axis=1)
    y = df['Group']

    print(f"[Data Loader] OASIS Alzheimer's Dataset Loaded. Shape: {X.shape}")
    return X, y