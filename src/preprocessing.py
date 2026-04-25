import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessing_pipeline(X_train: pd.DataFrame) -> ColumnTransformer:
    """Build a preprocessing pipeline for numeric and categorical features.

    Args:
        X_train: Training features DataFrame.

    Returns:
        ColumnTransformer with numeric and categorical preprocessing steps.
    """
    numeric_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X_train.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            # Use "Unknown" instead of mode to avoid demographic bias.
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    return preprocessor
