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


def load_parkinsons_v2(filepath='data/raw/pd_speech_features.csv'):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}.")

    df = pd.read_csv(filepath)

    # keep only the first recording to ensure no patient appears in both train and test.
    df = df.drop_duplicates(subset=['id'], keep='first').copy()

    #  drop for any rows with missing or corrupted audio parsing values.
    df = df.dropna()
    df = df.drop(columns=['id'])

    # split features
    X = df.drop('class', axis=1)
    y = df['class']

    print(f"Loaded Parkinson's (Sakar). Shape: {X.shape}")
    print(f"[Data Loader] Parkinson's Dataset Loaded. Shape: {X.shape}")
    return X, y

def load_autism(filepath='data/raw/autism_screening.csv'):
    """
    Loads and cleans the Autism Screening dataset.
    Target Variable: 'Class/ASD' (1 = YES, 0 = NO)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}.")

    df = pd.read_csv(filepath)

    # headers
    df.columns = df.columns.str.strip()

    # maps target yes/no to 1/0
    if 'Class/ASD' in df.columns:
        df['Class/ASD'] = df['Class/ASD'].astype(str).str.upper().map({'YES': 1, 'NO': 0})

    # result would just be the sum of 1-10, dropping so model doesn't cheat
    cols_to_drop = ['result', 'age_desc']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # incase the dataset uses '?' for missing values
    df = df.replace('?', pd.NA)
    df = df.dropna()

    # mapping binary columns
    binary_cols = ['jundice', 'autism', 'used_app_before']
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.upper().map({'YES': 1, 'NO': 0})
            
    if 'gender' in df.columns:
        df['gender'] = df['gender'].astype(str).str.upper().map({'M': 1, 'F': 0})

    # dummies for nominal categories
    df = pd.get_dummies(df, drop_first=True)

    # isolating feature(x) and target(y)
    X = df.drop('Class/ASD', axis=1)
    y = df['Class/ASD']

    print(f"[Data Loader] Autism Screening Dataset Loaded. Shape: {X.shape}")
    return X, y