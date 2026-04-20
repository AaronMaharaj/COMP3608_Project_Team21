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


def train_evaluate_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    seed: int = 67,
) -> Tuple[Pipeline, float, Dict[str, Any]]:
    """Train and evaluate a logistic regression model with hyperparameter tuning.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_test: Test features.
        y_test: Test labels.
        seed: Random seed (default: 67).

    Returns:
        Tuple of (model_pipeline, accuracy, classification_report_dict).
    """
    preprocessor = build_preprocessing_pipeline(X_train)

    # n_iter=10 keeps the search fast enough for the outer 5-fold loop.
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
        "clf__C": loguniform(1e-3, 1e1),  # Exponential search (1e-3 to 10).
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

    macro_avg = cast(Dict[str, float], rep.get("macro avg", {}))
    f1_score = macro_avg.get("f1-score", 0.0)
    precision = macro_avg.get("precision", 0.0)
    recall = macro_avg.get("recall", 0.0)

    print(
        f"   [LR]  Acc: {acc:.4f} | F1: {f1_score:.4f} | "
        f"Prec: {precision:.4f} | Rec: {recall:.4f}"
    )

    return best_pipeline, acc, rep


def train_evaluate_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    seed: int = 67,
) -> Tuple[Pipeline, float, Dict[str, Any]]:
    """Train and evaluate a random forest model with hyperparameter tuning.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_test: Test features.
        y_test: Test labels.
        seed: Random seed (default: 67).

    Returns:
        Tuple of (model_pipeline, accuracy, classification_report_dict).
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

    macro_avg = cast(Dict[str, float], rep.get("macro avg", {}))
    f1_score = macro_avg.get("f1-score", 0.0)
    precision = macro_avg.get("precision", 0.0)
    recall = macro_avg.get("recall", 0.0)

    print(
        f"   [RF]  Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f} | "
        f"F1: {f1_score:.4f} | "
        f"Prec: {precision:.4f} | Rec: {recall:.4f}"
    )

    return rf_model, test_acc, rep
