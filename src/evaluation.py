from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold, StratifiedKFold

from src.models_sklearn import train_evaluate_sklearn
from src.model_pytorch import train_evaluate_pytorch_model
from src.pipeline_factory import build_lr_search, build_rf_search


def _extract_macro_metrics(report: Dict[str, Any]) -> Dict[str, float]:
    """Extract f1, precision, recall from a classification_report dict."""
    macro = report.get("macro avg", {})
    return {
        "f1": float(macro.get("f1-score", 0.0)),
        "prec": float(macro.get("precision", 0.0)),
        "rec": float(macro.get("recall", 0.0)),
    }


def evaluate_pipeline(
    X: pd.DataFrame,
    y: pd.Series,
    groups: Optional[pd.Series] = None,
    n_splits: int = 5,
    fnn_threshold: float = 0.35,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, List[Dict[str, float]]]]:
    """Evaluate LR, RF, and FNN using cross-validation.

    Uses GroupKFold when ``groups`` is provided, StratifiedKFold otherwise.
    No side effects — does not print or save models.

    Args:
        X: Feature matrix.
        y: Target variable.
        groups: Patient/subject ID series for group splitting (optional).
        n_splits: Number of cross-validation folds (default: 5).
        fnn_threshold: Classification threshold for FNN (default: 0.35).

    Returns:
        Tuple of (summary_dict, fold_metrics_dict) where:
            - summary_dict maps model name to mean/std metrics.
            - fold_metrics_dict maps model name to per-fold metric dicts.
    """
    if groups is not None:
        splitter = GroupKFold(n_splits=n_splits)
        split_args: tuple = (X, y, groups)
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=67)
        split_args = (X, y)

    metrics: Dict[str, List[Dict[str, float]]] = {"LR": [], "RF": [], "FNN": []}

    for fold, (train_idx, test_idx) in enumerate(splitter.split(*split_args), 1):
        # Enforce strict determinism per-fold.
        torch.manual_seed(67 + fold)
        np.random.seed(67 + fold)

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        y_train = y_train.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)

        # Logistic Regression.
        lr_search = build_lr_search(X_train)
        lr_model, lr_acc, lr_rep = train_evaluate_sklearn(
            lr_search, X_train, y_train, X_test, y_test
        )
        lr_m = _extract_macro_metrics(lr_rep)
        metrics["LR"].append({"acc": lr_acc, **lr_m})

        # Random Forest.
        rf_search = build_rf_search(X_train)
        rf_model, rf_acc, rf_rep = train_evaluate_sklearn(
            rf_search, X_train, y_train, X_test, y_test
        )
        rf_m = _extract_macro_metrics(rf_rep)
        metrics["RF"].append({"acc": rf_acc, **rf_m})

        # Feed-Forward Neural Network (owns its own preprocessing).
        fnn_model, fnn_preproc, fnn_acc, fnn_rep = train_evaluate_pytorch_model(
            X_train, y_train, X_test, y_test, threshold=fnn_threshold
        )
        fnn_m = _extract_macro_metrics(fnn_rep)
        metrics["FNN"].append({"acc": fnn_acc, **fnn_m})

    # Aggregate mean ± std across folds.
    summary: Dict[str, Dict[str, float]] = {}
    for model_name, fold_metrics in metrics.items():
        summary[model_name] = {
            "Accuracy": float(np.mean([m["acc"] for m in fold_metrics])),
            "Std_Accuracy": float(np.std([m["acc"] for m in fold_metrics])),
            "F1-Score": float(np.mean([m["f1"] for m in fold_metrics])),
            "Std_F1_Score": float(np.std([m["f1"] for m in fold_metrics])),
            "Precision": float(np.mean([m["prec"] for m in fold_metrics])),
            "Std_Precision": float(np.std([m["prec"] for m in fold_metrics])),
            "Recall": float(np.mean([m["rec"] for m in fold_metrics])),
            "Std_Recall": float(np.std([m["rec"] for m in fold_metrics])),
        }

    return summary, metrics
