import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
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

    # Split categorical features into low and high cardinality
    low_cardinality_cols = [c for c in categorical_cols if X_train[c].nunique() <= 10]
    high_cardinality_cols = [c for c in categorical_cols if c not in low_cardinality_cols]

    numeric_transformer = Pipeline(
        steps=[
            # KNNImputer preserves non-linear relationships between SES, Education,
            # and Age — recommended over median imputation for clinical datasets.
            ("imputer", KNNImputer(n_neighbors=5)),
            ("scaler", StandardScaler()),
        ]
    )

    low_cat_transformer = Pipeline(
        steps=[
            # Use "Unknown" instead of mode to avoid demographic bias.
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    high_cat_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            (
                "onehot",
                OneHotEncoder(
                    min_frequency=0.05,
                    max_categories=10,
                    handle_unknown="infrequent_if_exist",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("low_cat", low_cat_transformer, low_cardinality_cols),
            ("high_cat", high_cat_transformer, high_cardinality_cols),
        ]
    )

    return preprocessor
