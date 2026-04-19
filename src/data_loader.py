import pandas as pd
import os


def load_alzheimers(filepath='data/raw/oasis_longitudinal.csv'):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}.")

    df = pd.read_csv(filepath)
    
    # Use only baseline visits to prevent the same patient appearing in train and test splits
    df = df.loc[df['Visit'] == 1].copy()
    
    # Enforce strict binary classification by excluding 'Converted' patients.
    # These patients progressed over time and create label ambiguity that confuses the model.
    df = df[df['Group'].isin(['Demented', 'Nondemented'])].copy()
    df['Group'] = df['Group'].map({'Demented': 1, 'Nondemented': 0})
    df['M/F'] = df['M/F'].map({'M': 1, 'F': 0})

    # Remove identifiers and columns that cause target leakage (CDR is directly used to assign Group)
    cols_to_drop = ['Subject ID', 'MRI ID', 'Hand', 'CDR', 'Visit', 'MR Delay']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    df = df.dropna(subset=['Group'])

    X = df.drop('Group', axis=1)
    y = df['Group'].astype(int)

    print(f"Loaded OASIS Alzheimer's. Shape: {X.shape}")
    return X, y


def load_parkinsons_v2(filepath='data/raw/pd_speech_features.csv'):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}.")

    df = pd.read_csv(filepath)

    # Sort by 'id' to guarantee we keep the baseline recordings deterministically,
    # then keep only the first recording to ensure no patient appears in both train and test.
    df = df.sort_values('id').drop_duplicates(subset=['id'], keep='first').copy()

    #  drop for any rows with missing or corrupted audio parsing values.
    df = df.dropna()
    df = df.drop(columns=['id'])

    # split features
    X = df.drop('class', axis=1)
    y = df['class']

    print(f"Loaded Parkinson's (Sakar). Shape: {X.shape}")
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

    # Drop AQ-10 item scores to ensure the model uses demographic and behavioural predictors.
    # We also drop 'result' because it is literally the sum of A1-A10 (direct target leakage).
    aq10_items = [f'A{i}_Score' for i in range(1, 11)]
    cols_to_drop = ['result', 'age_desc'] + aq10_items
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # incase the dataset uses '?' for missing values
    df = df.replace('?', pd.NA)
    before_drop = len(df)
    df = df.dropna()
    dropped_count = before_drop - len(df)
    if dropped_count > 0:
        print(f"   [Autism] Dropped {dropped_count} rows with missing values.")

    # isolating feature(x) and target(y)
    X = df.drop('Class/ASD', axis=1)
    y = df['Class/ASD']

    print(f"[Data Loader] Autism Screening Dataset Loaded. Shape: {X.shape}")
    return X, y