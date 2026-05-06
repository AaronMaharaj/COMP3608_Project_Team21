from typing import Union

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from scipy.stats import loguniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import BaseCrossValidator, RandomizedSearchCV

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
            ("smote", SMOTE(random_state=seed, k_neighbors=3)),
            (
                "clf",
                LogisticRegression(
                    max_iter=10000,
                    random_state=seed,
                    class_weight="balanced",
                    solver="saga",  # saga supports elasticnet / l1 / l2
                ),
            ),
        ]
    )

    param_dist = [
        {
            "clf__penalty": ["elasticnet"],
            "clf__C": loguniform(1e-3, 1e1),
            # l1_ratio=1.0 → pure L1 (Lasso), 0.0 → pure L2 (Ridge)
            "clf__l1_ratio": [0.0, 0.25, 0.5, 0.75, 1.0],
        },
        {
            "clf__penalty": ["l1", "l2"],
            "clf__C": loguniform(1e-3, 1e1),
        }
    ]

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
    n_iter: int = 20,
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
            ("smote", SMOTE(random_state=seed, k_neighbors=3)),
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
        "clf__n_estimators": [100, 200, 300, 500, 700, 1000],
        "clf__max_depth": [5, 10, 15, 20, 30, None],
        "clf__min_samples_split": [2, 5, 10],
        "clf__min_samples_leaf": [2, 3, 5],
        "clf__max_features": ["sqrt", "log2"],
        "clf__criterion": ["gini", "entropy", "log_loss"],
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
