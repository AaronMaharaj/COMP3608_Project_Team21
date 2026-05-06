from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    fbeta_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    StratifiedKFold,
    train_test_split,
)

from src.model_pytorch import train_evaluate_pytorch_model
from src.models_sklearn import _extract_positive_class_scores, train_evaluate_sklearn
from src.pipeline_factory import build_lr_search, build_rf_search


def _extract_macro_metrics(report: Dict[str, Any]) -> Dict[str, float]:
    """Extract f1, precision, recall from a classification_report dict."""
    macro = report.get("macro avg", {})
    return {
        "f1": float(macro.get("f1-score", 0.0)),
        "prec": float(macro.get("precision", 0.0)),
        "rec": float(macro.get("recall", 0.0)),
    }


def _safe_auc_roc(y_true: pd.Series, y_score: np.ndarray) -> float:
    """Compute binary ROC-AUC safely.

    Returns NaN when a fold lacks both classes or when AUC cannot be computed.
    """
    y_true_arr = np.asarray(y_true)
    y_score_arr = np.asarray(y_score, dtype=float).reshape(-1)

    if np.unique(y_true_arr).size < 2:
        return float("nan")

    try:
        return float(roc_auc_score(y_true_arr, y_score_arr))
    except ValueError:
        return float("nan")


def _mean_std_ignore_nan(values: List[float]) -> Tuple[float, float]:
    """Return mean and std while ignoring NaNs; all-NaN yields NaN/NaN."""
    arr = np.asarray(values, dtype=float)
    valid = arr[~np.isnan(arr)]
    if valid.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(valid)), float(np.std(valid))


def _carve_validation_slice(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: Optional[pd.Series],
    test_size: float = 0.1,
    max_attempts: int = 15,
    seed: int = 67,
) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Optional[pd.Series]]]:
    """Carve a validation slice from a training fold for early-stop and threshold tuning.

    For grouped data, uses GroupShuffleSplit with retry to guarantee a slice that
    is (a) group-disjoint, (b) class-balanced in val, (c) class-balanced in sub-train.
    For ungrouped data, uses stratified train_test_split.

    Returns (X_sub, X_val, y_sub, y_val, groups_sub) on success, or None if all
    retry attempts fail (caller should fall back to a fixed-threshold path).
    """
    if groups_train is not None:
        for attempt in range(max_attempts):
            gss = GroupShuffleSplit(
                n_splits=1, test_size=test_size, random_state=seed + attempt
            )
            train_idx_sub, val_idx = next(
                gss.split(X_train, y_train, groups=groups_train)
            )

            X_sub = X_train.iloc[train_idx_sub]
            X_val = X_train.iloc[val_idx]
            y_sub = y_train.iloc[train_idx_sub]
            y_val = y_train.iloc[val_idx]

            train_groups = set(groups_train.iloc[train_idx_sub])
            val_groups = set(groups_train.iloc[val_idx])
            if train_groups.intersection(val_groups):
                continue
            if len(set(y_val)) < 2:
                continue
            if len(set(y_sub)) < 2:
                continue

            groups_sub = groups_train.iloc[train_idx_sub].reset_index(drop=True)
            return (
                X_sub.reset_index(drop=True),
                X_val.reset_index(drop=True),
                y_sub.reset_index(drop=True),
                y_val.reset_index(drop=True),
                groups_sub,
            )
        return None

    X_sub, X_val, y_sub, y_val = train_test_split(
        X_train, y_train, test_size=test_size, stratify=y_train, random_state=seed
    )
    return (
        X_sub.reset_index(drop=True),
        X_val.reset_index(drop=True),
        y_sub.reset_index(drop=True),
        y_val.reset_index(drop=True),
        None,
    )


def _metrics_at_threshold(
    y_true: np.ndarray, scores: np.ndarray, t: float
) -> Dict[str, float]:
    """Compute (balanced_recall, youden_j, sensitivity, specificity, f2) at threshold t."""
    preds = (scores >= t).astype(int)
    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sens = float(tp) / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = float(tn) / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        "threshold": float(t),
        "balanced_recall": 0.5 * (sens + spec),
        "youden_j": sens + spec - 1.0,
        "sensitivity": sens,
        "specificity": spec,
        "f2": float(fbeta_score(y_true, preds, beta=2, zero_division=0)),
    }


def _sweep_threshold(
    y_val: np.ndarray, scores: np.ndarray
) -> Dict[str, float]:
    """Pick threshold maximising balanced recall (= recall_macro = Youden + 0.5).

    Sweep is exact over unique score boundaries when small; otherwise falls back
    to a linspace grid. Tie-break: pick threshold closest to 0.5 among the maximisers.
    On total degeneracy, returns the metrics at t=0.5.
    """
    y_arr = np.asarray(y_val).astype(int)
    s_arr = np.asarray(scores, dtype=float).reshape(-1)

    uniq = np.unique(s_arr)
    if 1 < len(uniq) <= 500:
        thresholds = uniq
    else:
        thresholds = np.linspace(0.01, 0.99, 99)

    best_score = -np.inf
    candidates: List[float] = []
    for t in thresholds:
        preds = (s_arr >= t).astype(int)
        # Skip degenerate splits where the model assigns one class only and val
        # has both classes — balanced_accuracy collapses to 0.5 and never wins.
        if len(np.unique(preds)) < 2 and len(np.unique(y_arr)) == 2:
            continue
        ba = balanced_accuracy_score(y_arr, preds)
        if ba > best_score + 1e-9:
            best_score = ba
            candidates = [float(t)]
        elif abs(ba - best_score) <= 1e-9:
            candidates.append(float(t))

    if not candidates:
        return _metrics_at_threshold(y_arr, s_arr, t=0.5)

    t_star = float(min(candidates, key=lambda x: abs(x - 0.5)))
    return _metrics_at_threshold(y_arr, s_arr, t=t_star)


def evaluate_pipeline(
    X: pd.DataFrame,
    y: pd.Series,
    groups: Optional[pd.Series] = None,
    n_splits: int = 5,
    fnn_threshold: float = 0.35,
    run_threshold_sweep: bool = True,
) -> Tuple[
    Dict[str, Dict[str, float]],
    Dict[str, List[Dict[str, float]]],
    Dict[str, List[float]],
]:
    """Evaluate LR, RF, and FNN using cross-validation.

    Uses GroupKFold when ``groups`` is provided, StratifiedKFold otherwise. When
    ``run_threshold_sweep`` is True, carves a group-aware validation slice from
    each outer training fold, fits all three models on the sub-train, and selects
    the per-model decision threshold that maximises balanced recall on the val
    slice. The chosen threshold is then applied to the outer test fold.

    When ``run_threshold_sweep`` is False, falls back to fixed thresholds (0.5 for
    sklearn models via predict(), ``fnn_threshold`` for the FNN). This path is
    used for datasets with deterministic labels where threshold sweeping is
    uninformative (e.g. Autism Screening where Class/ASD is a deterministic
    function of A1-A10).

    Inner-CV scoring inside RandomizedSearchCV remains ``recall_macro`` (see
    src/pipeline_factory.py); the threshold sweep uses balanced_accuracy_score
    which is the binary equivalent of recall_macro — coherent end-to-end.

    Args:
        X: Feature matrix.
        y: Target variable.
        groups: Patient/subject ID series for group splitting (optional).
        n_splits: Number of cross-validation folds (default: 5).
        fnn_threshold: FNN threshold used on the no-sweep path (default: 0.35).
        run_threshold_sweep: When True, carve+sweep per fold; when False, use
            fixed thresholds throughout (default: True).

    Returns:
        Tuple (summary_dict, fold_metrics_dict, per_fold_thresholds) where:
            - summary_dict maps model name to mean/std metrics including new
              sweep-derived columns (Threshold_Median, Sensitivity, Specificity,
              Balanced_Recall, Youden_J, F2_Score).
            - fold_metrics_dict maps model name to per-fold metric dicts.
            - per_fold_thresholds maps model name to the list of fold thresholds
              (used by main.py to derive the deployment threshold).
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

        inner_cv: Any = 3
        groups_train: Optional[pd.Series] = None

        if groups is not None:
            train_groups = set(groups.iloc[train_idx])
            test_groups = set(groups.iloc[test_idx])
            overlap = train_groups.intersection(test_groups)
            if overlap:
                raise ValueError(
                    f"Outer split leakage detected: {len(overlap)} groups overlap between train and test."
                )

            inner_cv = GroupKFold(n_splits=3)
            groups_train = groups.iloc[train_idx].reset_index(drop=True)

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        X_train = X_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)

        # ------------------------------------------------------------------
        # Shared validation carve-out.
        # When the sweep is enabled, all three models are fit on the same
        # sub-train and pick their threshold against the same val slice.
        # ------------------------------------------------------------------
        carve = None
        if run_threshold_sweep:
            carve = _carve_validation_slice(
                X_train, y_train, groups_train, seed=67 + fold
            )
            if carve is None:
                print(
                    f"  [WARN] Fold {fold}: validation carve failed after retries; "
                    f"falling back to fixed thresholds for this fold."
                )

        if carve is not None:
            X_sub, X_val, y_sub, y_val, groups_sub = carve
            fit_X, fit_y, fit_groups = X_sub, y_sub, groups_sub
        else:
            X_val = y_val = None
            fit_X, fit_y, fit_groups = X_train, y_train, groups_train

        # ------------------------------------------------------------------
        # Logistic Regression.
        # ------------------------------------------------------------------
        lr_search = build_lr_search(fit_X, cv=inner_cv)
        lr_model, _, _, lr_test_scores, lr_val_scores = train_evaluate_sklearn(
            lr_search,
            fit_X,
            fit_y,
            X_test,
            y_test,
            groups_train=fit_groups,
            X_val=X_val,
        )

        if lr_val_scores is not None and y_val is not None:
            lr_sweep = _sweep_threshold(np.asarray(y_val), lr_val_scores)
        else:
            lr_sweep = _metrics_at_threshold(np.asarray(y_test), lr_test_scores, t=0.5)

        lr_t = lr_sweep["threshold"]
        lr_preds = (lr_test_scores >= lr_t).astype(int)
        lr_acc = float(accuracy_score(y_test, lr_preds))
        lr_rep = classification_report(
            y_test, lr_preds, zero_division=0, output_dict=True
        )
        lr_m = _extract_macro_metrics(lr_rep)
        lr_auc = _safe_auc_roc(y_test, lr_test_scores)
        metrics["LR"].append({"acc": lr_acc, **lr_m, "auc": lr_auc, **lr_sweep})

        # ------------------------------------------------------------------
        # Random Forest.
        # ------------------------------------------------------------------
        rf_search = build_rf_search(fit_X, cv=inner_cv)
        rf_model, _, _, rf_test_scores, rf_val_scores = train_evaluate_sklearn(
            rf_search,
            fit_X,
            fit_y,
            X_test,
            y_test,
            groups_train=fit_groups,
            X_val=X_val,
        )

        if rf_val_scores is not None and y_val is not None:
            rf_sweep = _sweep_threshold(np.asarray(y_val), rf_val_scores)
        else:
            rf_sweep = _metrics_at_threshold(np.asarray(y_test), rf_test_scores, t=0.5)

        rf_t = rf_sweep["threshold"]
        rf_preds = (rf_test_scores >= rf_t).astype(int)
        rf_acc = float(accuracy_score(y_test, rf_preds))
        rf_rep = classification_report(
            y_test, rf_preds, zero_division=0, output_dict=True
        )
        rf_m = _extract_macro_metrics(rf_rep)
        rf_auc = _safe_auc_roc(y_test, rf_test_scores)
        metrics["RF"].append({"acc": rf_acc, **rf_m, "auc": rf_auc, **rf_sweep})

        # ------------------------------------------------------------------
        # Feed-Forward Neural Network.
        # If the shared carve produced a val slice, hand it to the FNN so it
        # skips its internal 15-attempt retry. Otherwise the FNN re-carves
        # internally (legacy behaviour preserved for the no-sweep path).
        # ------------------------------------------------------------------
        fnn_call_threshold = fnn_threshold if not run_threshold_sweep else 0.5
        (
            fnn_model,
            fnn_preproc,
            fnn_acc_initial,
            fnn_rep_initial,
            fnn_test_scores,
            fnn_val_scores,
        ) = train_evaluate_pytorch_model(
            fit_X,
            fit_y,
            X_test,
            y_test,
            threshold=fnn_call_threshold,
            groups_train=fit_groups,
            X_val_precarved=X_val,
            y_val_precarved=y_val,
        )

        if (
            run_threshold_sweep
            and fnn_val_scores is not None
            and y_val is not None
        ):
            fnn_sweep = _sweep_threshold(np.asarray(y_val), fnn_val_scores)
        else:
            fnn_sweep = _metrics_at_threshold(
                np.asarray(y_test), fnn_test_scores, t=fnn_call_threshold
            )

        fnn_t = fnn_sweep["threshold"]
        fnn_preds = (fnn_test_scores >= fnn_t).astype(int)
        fnn_acc = float(accuracy_score(y_test, fnn_preds))
        fnn_rep = classification_report(
            y_test, fnn_preds, zero_division=0, output_dict=True
        )
        fnn_m = _extract_macro_metrics(fnn_rep)
        fnn_auc = _safe_auc_roc(y_test, fnn_test_scores)
        metrics["FNN"].append(
            {"acc": fnn_acc, **fnn_m, "auc": fnn_auc, **fnn_sweep}
        )

    # ------------------------------------------------------------------
    # Aggregate mean ± std across folds. New sweep-derived columns are
    # appended to the legacy schema; existing consumers keep working.
    # ------------------------------------------------------------------
    summary: Dict[str, Dict[str, float]] = {}
    per_fold_thresholds: Dict[str, List[float]] = {}
    for model_name, fold_metrics in metrics.items():
        auc_mean, auc_std = _mean_std_ignore_nan([m["auc"] for m in fold_metrics])
        thresholds_arr = np.asarray([m["threshold"] for m in fold_metrics])
        summary[model_name] = {
            "Accuracy": float(np.mean([m["acc"] for m in fold_metrics])),
            "Std_Accuracy": float(np.std([m["acc"] for m in fold_metrics])),
            "F1-Score": float(np.mean([m["f1"] for m in fold_metrics])),
            "Std_F1_Score": float(np.std([m["f1"] for m in fold_metrics])),
            "Precision": float(np.mean([m["prec"] for m in fold_metrics])),
            "Std_Precision": float(np.std([m["prec"] for m in fold_metrics])),
            "Recall": float(np.mean([m["rec"] for m in fold_metrics])),
            "Std_Recall": float(np.std([m["rec"] for m in fold_metrics])),
            "AUC-ROC": auc_mean,
            "Std_AUC_ROC": auc_std,
            "Threshold_Median": float(np.median(thresholds_arr)),
            "Threshold_Std": float(np.std(thresholds_arr)),
            "Balanced_Recall": float(
                np.mean([m["balanced_recall"] for m in fold_metrics])
            ),
            "Std_Balanced_Recall": float(
                np.std([m["balanced_recall"] for m in fold_metrics])
            ),
            "Youden_J": float(np.mean([m["youden_j"] for m in fold_metrics])),
            "Sensitivity": float(np.mean([m["sensitivity"] for m in fold_metrics])),
            "Specificity": float(np.mean([m["specificity"] for m in fold_metrics])),
            "F2_Score": float(np.mean([m["f2"] for m in fold_metrics])),
        }
        per_fold_thresholds[model_name] = thresholds_arr.tolist()

        if float(np.std(thresholds_arr)) > 0.15 and run_threshold_sweep:
            print(
                f"  [WARN] {model_name}: per-fold threshold std="
                f"{float(np.std(thresholds_arr)):.3f} is high. "
                f"Per-fold values: {thresholds_arr.tolist()}"
            )

    return summary, metrics, per_fold_thresholds
