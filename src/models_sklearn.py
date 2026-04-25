from typing import Any, Dict, Tuple, cast

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline


def train_evaluate_sklearn(
    search: RandomizedSearchCV,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[Pipeline, float, Dict[str, Any]]:
    """Fit a RandomizedSearchCV and evaluate on a held-out test set.

    Args:
        search: Configured, unfitted RandomizedSearchCV (from pipeline_factory).
        X_train: Training features.
        y_train: Training labels.
        X_test: Test features.
        y_test: Test labels.

    Returns:
        Tuple of (best_pipeline, test_accuracy, classification_report_dict).
    """
    search.fit(X_train, y_train)
    best_pipeline = cast(Pipeline, search.best_estimator_)

    predictions = best_pipeline.predict(X_test)
    acc = float(accuracy_score(y_test, predictions))
    rep = cast(
        Dict[str, Any],
        classification_report(y_test, predictions, zero_division=0, output_dict=True),
    )

    return best_pipeline, acc, rep
