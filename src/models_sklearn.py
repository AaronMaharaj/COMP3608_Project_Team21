from typing import Any, Dict, Optional, Tuple, cast

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline


def _extract_positive_class_scores(model: Pipeline, X_test: pd.DataFrame) -> np.ndarray:
    """Return continuous scores for the positive class (label=1) for AUC.

    Prefers ``predict_proba`` and falls back to ``decision_function`` when
    probability estimates are unavailable.
    """
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)
        if probs.ndim == 1:
            return probs.astype(float)

        classes = getattr(model, "classes_", None)
        if classes is not None and len(classes) == probs.shape[1] and 1 in classes:
            pos_idx = int(np.where(np.array(classes) == 1)[0][0])
        else:
            pos_idx = 1 if probs.shape[1] > 1 else 0
        return probs[:, pos_idx].astype(float)

    if hasattr(model, "decision_function"):
        decision = np.asarray(model.decision_function(X_test), dtype=float)
        if decision.ndim == 1:
            return decision
        pos_idx = 1 if decision.shape[1] > 1 else 0
        return decision[:, pos_idx]

    # Last-resort fallback keeps evaluation pipeline resilient.
    return np.asarray(model.predict(X_test), dtype=float)


def train_evaluate_sklearn(
    search: RandomizedSearchCV,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    groups_train: Optional[pd.Series] = None,
    X_val: Optional[pd.DataFrame] = None,
    threshold: float = 0.5,
) -> Tuple[Pipeline, float, Dict[str, Any], np.ndarray, Optional[np.ndarray]]:
    """Fit a RandomizedSearchCV and evaluate on a held-out test set.

    When ``X_val`` is provided, also returns positive-class scores on the
    validation slice so the caller can run a threshold sweep. Threshold
    application on the test set is performed here via
    ``predict_proba >= threshold`` (bypassing sklearn's hardcoded 0.5 in
    ``predict()``); the caller is responsible for selecting ``threshold``
    upstream of this function.

    Args:
        search: Configured, unfitted RandomizedSearchCV (from pipeline_factory).
        X_train: Training features.
        y_train: Training labels.
        X_test: Test features.
        y_test: Test labels.
        groups_train: Optional training groups for inner CV.
        X_val: Optional validation features to score for threshold sweeping.
        threshold: Decision threshold applied to predict_proba on the test set
            (default: 0.5).

    Returns:
        Tuple (best_pipeline, test_accuracy, classification_report_dict,
        test_positive_class_scores, val_positive_class_scores_or_None).
    """
    if groups_train is not None:
        search.fit(X_train, y_train, groups=groups_train)
    else:
        search.fit(X_train, y_train)
    best_pipeline = cast(Pipeline, search.best_estimator_)

    test_scores = _extract_positive_class_scores(best_pipeline, X_test)
    val_scores: Optional[np.ndarray] = None
    if X_val is not None:
        val_scores = _extract_positive_class_scores(best_pipeline, X_val)

    predictions = (test_scores >= threshold).astype(int)
    acc = float(accuracy_score(y_test, predictions))
    rep = cast(
        Dict[str, Any],
        classification_report(y_test, predictions, zero_division=0, output_dict=True),
    )

    return best_pipeline, acc, rep, test_scores, val_scores
