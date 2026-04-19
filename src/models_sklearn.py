from typing import Any, Dict, Tuple, cast

import pandas as pd

from scipy.stats import loguniform
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessing_pipeline(X_train: pd.DataFrame) -> ColumnTransformer:
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
            ("imputer", SimpleImputer(strategy="most_frequent")),
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


def train_evaluate_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    seed: int = 67,
) -> Tuple[Pipeline, float, Dict[str, Any]]:
    preprocessor = build_preprocessing_pipeline(X_train)

    # n_iter=10 keeps the search fast enough for the outer 5-fold loop
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000, random_state=seed, class_weight="balanced"
                ),
            ),
        ]
    )
    param_dist = {
        "clf__C": loguniform(1e-3, 1e1),
        "clf__solver": ["lbfgs", "saga"],
    }
    search = RandomizedSearchCV(
        pipeline,
        param_dist,
        n_iter=10,
        cv=3,
        scoring="f1_macro",
        random_state=seed,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    best_pipeline = cast(Pipeline, search.best_estimator_)

    predictions = best_pipeline.predict(X_test)
    acc = float(accuracy_score(y_test, predictions))
    rep = cast(
        Dict[str, Any],
        classification_report(y_test, predictions, zero_division=0, output_dict=True),
    )

    print(
        f"   [LR]  Acc: {acc:.4f} | F1: {rep['macro avg']['f1-score']:.4f} | "
        f"Prec: {rep['macro avg']['precision']:.4f} | Rec: {rep['macro avg']['recall']:.4f}"
    )
    return best_pipeline, acc, rep


def train_evaluate_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    seed: int = 67,
) -> Tuple[Pipeline, float, Dict[str, Any]]:
    preprocessor = build_preprocessing_pipeline(X_train)

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "clf",
                RandomForestClassifier(
                    random_state=seed, n_jobs=1, class_weight="balanced"
                ),
            ),
        ]
    )

    # default backend can deadlock on windows, n_jobs=1, -1 search to avoid
    param_dist_rf = {
        "clf__n_estimators": [100, 150, 200, 300],
        "clf__max_depth": [5, 7, 10, None],
        "clf__min_samples_leaf": [2, 3, 5],
    }

    search_rf = RandomizedSearchCV(
        pipeline,
        param_dist_rf,
        n_iter=10,
        cv=3,
        scoring="f1_macro",
        random_state=seed,
        n_jobs=-1,
    )
    search_rf.fit(X_train, y_train)
    rf_model = cast(Pipeline, search_rf.best_estimator_)

    train_preds = rf_model.predict(X_train)
    test_preds = rf_model.predict(X_test)

    train_acc = float(accuracy_score(y_train, train_preds))
    test_acc = float(accuracy_score(y_test, test_preds))
    rep = cast(
        Dict[str, Any],
        classification_report(y_test, test_preds, zero_division=0, output_dict=True),
    )

    print(
        f"   [RF]  Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f} | "
        f"F1: {rep['macro avg']['f1-score']:.4f} | "
        f"Prec: {rep['macro avg']['precision']:.4f} | Rec: {rep['macro avg']['recall']:.4f}"
    )

    return rf_model, test_acc, rep
