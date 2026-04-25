from typing import Union

import pandas as pd
from scipy.stats import loguniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import BaseCrossValidator, RandomizedSearchCV
from sklearn.pipeline import Pipeline

from src.preprocessing import build_preprocessing_pipeline


def build_lr_search(
    X_train: pd.DataFrame,
    cv: Union[int, BaseCrossValidator] = 3,
    n_iter: int = 10,
    seed: int = 67,
) -> RandomizedSearchCV:
    """Return a configured, unfitted RandomizedSearchCV for Logistic Regression.

    Args:
        X_train: Training features (used to infer column types for preprocessing).
        cv: Cross-validation strategy for inner hyperparameter tuning.
        n_iter: Number of random search iterations.
        seed: Random seed.

    Returns:
        Unfitted RandomizedSearchCV wrapping a Pipeline.
    """
    preprocessor = build_preprocessing_pipeline(X_train)
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "clf",
                LogisticRegression(
                    max_iter=10000, random_state=seed, class_weight="balanced"
                ),
            ),
        ]
    )

    param_dist = {
        "clf__C": loguniform(1e-3, 1e1),
        "clf__solver": ["lbfgs", "saga"],
    }

    return RandomizedSearchCV(
        pipeline,
        param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring="recall_macro",
        random_state=seed,
        n_jobs=-1,
    )


def build_rf_search(
    X_train: pd.DataFrame,
    cv: Union[int, BaseCrossValidator] = 3,
    n_iter: int = 10,
    seed: int = 67,
) -> RandomizedSearchCV:
    """Return a configured, unfitted RandomizedSearchCV for Random Forest.

    Args:
        X_train: Training features (used to infer column types for preprocessing).
        cv: Cross-validation strategy for inner hyperparameter tuning.
        n_iter: Number of random search iterations.
        seed: Random seed.

    Returns:
        Unfitted RandomizedSearchCV wrapping a Pipeline.
    """
    preprocessor = build_preprocessing_pipeline(X_train)
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "clf",
                RandomForestClassifier(
                    random_state=seed,
                    n_jobs=1,  # Avoid Windows deadlock with nested parallelism.
                    class_weight="balanced",
                ),
            ),
        ]
    )

    param_dist = {
        "clf__n_estimators": [100, 150, 200, 300],
        "clf__max_depth": [5, 7, 10, None],
        "clf__min_samples_leaf": [2, 3, 5],
    }

    return RandomizedSearchCV(
        pipeline,
        param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring="recall_macro",
        random_state=seed,
        n_jobs=-1,
    )
