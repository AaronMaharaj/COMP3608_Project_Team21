"""Render report figures from CV artifacts.

Reads:
    artifacts/cv_arrays/<dataset>__<model>.npz   (written by main.py)
    models/production/<dataset>_rf.joblib        (written by train_production.py)

Writes:
    figures/<dataset>__threshold_cost.png        (LR/RF/FNN, 3 panels)
    figures/<dataset>__pr_curves.png             (LR/RF/FNN overlaid)
    figures/<dataset>__confusion.png             (LR/RF/FNN, 3 panels)
    figures/<dataset>__rf_feature_importance.png (top 10 RF features)

Run after main.py (for the .npz files) and train_production.py (for the .joblib
files). Usage:
    python make_visuals.py
"""
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

from src.persistence import load_sklearn_model
from src.visualizations import (
    DEFAULT_C_FN,
    DEFAULT_C_FP,
    concat_folds,
    plot_calibration_curves,
    plot_cost_confusion_matrix,
    plot_dca_curves,
    plot_pr_curves,
    plot_rf_feature_importance,
    plot_roc_curves,
    plot_threshold_cost_curve,
)

ARTIFACTS_DIR = Path("artifacts") / "cv_arrays"
PRODUCTION_DIR = Path("models") / "production"
FIGURES_DIR = Path("figures")

DATASETS = ["Autism_Screening", "OASIS_Alzheimers", "Parkinsons_Sakar"]
DATASET_LABELS = {
    "Autism_Screening": "Autism Screening",
    "OASIS_Alzheimers": "OASIS Alzheimer's",
    "Parkinsons_Sakar": "Parkinson's (Sakar)",
}
MODELS = ["LR", "RF", "FNN"]


def _load_fold_list(dataset: str, model: str) -> Optional[List[Dict[str, np.ndarray]]]:
    """Re-hydrate the flat .npz back into a list-of-dicts."""
    path = ARTIFACTS_DIR / f"{dataset}__{model}.npz"
    if not path.exists():
        print(f"  [WARN] missing {path}; skipping.")
        return None
    with np.load(path) as npz:
        keys = list(npz.keys())
        n_folds = sum(1 for k in keys if k.endswith("_y_true"))
        folds: List[Dict[str, np.ndarray]] = []
        for i in range(n_folds):
            folds.append(
                {
                    "y_true": npz[f"fold{i}_y_true"],
                    "y_score": npz[f"fold{i}_y_score"],
                    "threshold": npz[f"fold{i}_threshold"],
                }
            )
        return folds


def _render_threshold_cost(dataset: str, fold_arrays: Dict[str, list]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, model in zip(axes, MODELS):
        folds = fold_arrays.get(model)
        if not folds:
            ax.set_visible(False)
            continue
        y_true, y_score, t_med = concat_folds(folds)
        plot_threshold_cost_curve(
            y_true, y_score, t_star=t_med, ax=ax, model_label=model,
            c_fn=DEFAULT_C_FN, c_fp=DEFAULT_C_FP,
        )
    fig.suptitle(
        f"{DATASET_LABELS[dataset]}: cost-sensitive threshold sweep "
        f"(C_FN={DEFAULT_C_FN:.0f}, C_FP={DEFAULT_C_FP:.0f})",
        fontsize=12,
    )
    fig.tight_layout()
    out = FIGURES_DIR / f"{dataset}__threshold_cost.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out}")


def _render_pr_curves(dataset: str, fold_arrays: Dict[str, list]) -> None:
    arrays_by_model: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for model in MODELS:
        folds = fold_arrays.get(model)
        if not folds:
            continue
        y_true, y_score, _ = concat_folds(folds)
        arrays_by_model[model] = (y_true, y_score)

    if not arrays_by_model:
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    plot_pr_curves(arrays_by_model, ax=ax, dataset_label=DATASET_LABELS[dataset])
    fig.tight_layout()
    out = FIGURES_DIR / f"{dataset}__pr_curves.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out}")


def _render_roc_curves(dataset: str, fold_arrays: Dict[str, list]) -> None:
    arrays_by_model: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for model in MODELS:
        folds = fold_arrays.get(model)
        if not folds:
            continue
        y_true, y_score, _ = concat_folds(folds)
        arrays_by_model[model] = (y_true, y_score)

    if not arrays_by_model:
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    plot_roc_curves(arrays_by_model, ax=ax, dataset_label=DATASET_LABELS[dataset])
    fig.tight_layout()
    out = FIGURES_DIR / f"{dataset}__roc_curves.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out}")


def _render_dca(dataset: str, fold_arrays: Dict[str, list]) -> None:
    arrays_by_model: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for model in MODELS:
        folds = fold_arrays.get(model)
        if not folds:
            continue
        y_true, y_score, _ = concat_folds(folds)
        arrays_by_model[model] = (y_true, y_score)

    if not arrays_by_model:
        return

    # Adapt p_t range to dataset prevalence: capping at min(prevalence + 0.1,
    # 0.6) keeps the chart focussed on the clinically relevant decision band.
    sample_y = next(iter(arrays_by_model.values()))[0]
    prevalence = float(sample_y.mean()) if len(sample_y) > 0 else 0.5
    p_t_max = max(0.3, min(prevalence + 0.1, 0.85))

    fig, ax = plt.subplots(figsize=(7, 6))
    plot_dca_curves(
        arrays_by_model, ax=ax, dataset_label=DATASET_LABELS[dataset],
        p_t_max=p_t_max,
    )
    fig.tight_layout()
    out = FIGURES_DIR / f"{dataset}__dca_curves.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out}")


def _render_calibration(dataset: str, fold_arrays: Dict[str, list]) -> None:
    arrays_by_model: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for model in MODELS:
        folds = fold_arrays.get(model)
        if not folds:
            continue
        y_true, y_score, _ = concat_folds(folds)
        arrays_by_model[model] = (y_true, y_score)

    if not arrays_by_model:
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    plot_calibration_curves(arrays_by_model, ax=ax, dataset_label=DATASET_LABELS[dataset])
    fig.tight_layout()
    out = FIGURES_DIR / f"{dataset}__calibration.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out}")


def _render_confusion(dataset: str, fold_arrays: Dict[str, list]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, model in zip(axes, MODELS):
        folds = fold_arrays.get(model)
        if not folds:
            ax.set_visible(False)
            continue
        y_true, y_score, t_med = concat_folds(folds)
        y_pred = (y_score >= t_med).astype(int)
        plot_cost_confusion_matrix(
            y_true, y_pred, ax=ax, model_label=f"{model} @ t={t_med:.2f}",
        )
    fig.suptitle(
        f"{DATASET_LABELS[dataset]}: confusion matrix at swept threshold",
        fontsize=12,
    )
    fig.tight_layout()
    out = FIGURES_DIR / f"{dataset}__confusion.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out}")


def _render_rf_feature_importance(dataset: str) -> None:
    rf_path = PRODUCTION_DIR / f"{dataset}_rf.joblib"
    if not rf_path.exists():
        print(f"  [WARN] missing {rf_path}; skipping RF importance plot.")
        return

    loaded, _, _ = load_sklearn_model(str(rf_path))

    # Production RF is wrapped in CalibratedClassifierCV (5-fold ensemble of
    # the fitted imblearn Pipeline). Average feature importances across the
    # ensemble's component pipelines for a stable bar chart.
    if hasattr(loaded, "calibrated_classifiers_"):
        component_pipelines = [
            cc.estimator for cc in loaded.calibrated_classifiers_
            if hasattr(cc, "estimator") and hasattr(cc.estimator, "named_steps")
        ]
        if not component_pipelines:
            print(f"  [WARN] {rf_path}: calibrator has no usable pipelines; skipping.")
            return
        # Each calibrated component was fit on a different CV slice, so their
        # post-encoding feature counts can differ slightly (OneHotEncoder
        # min_frequency varies with the slice). Use fold 0 as representative.
        rep = component_pipelines[0]
        preprocessor = rep.named_steps.get("preprocessor")
        selector = rep.named_steps.get("selector")
        clf = rep.named_steps.get("clf")
        if clf is None or not hasattr(clf, "feature_importances_"):
            print(f"  [WARN] {rf_path}: no feature importances on fold 0; skipping.")
            return
        importances = np.asarray(clf.feature_importances_, dtype=float)
    elif hasattr(loaded, "named_steps"):
        preprocessor = loaded.named_steps.get("preprocessor")
        clf = loaded.named_steps.get("clf")
        selector = loaded.named_steps.get("selector")
        if clf is None:
            print(f"  [WARN] unexpected pipeline shape in {rf_path}; skipping.")
            return
        importances = np.asarray(clf.feature_importances_, dtype=float)
    else:
        print(f"  [WARN] {rf_path} is not a Pipeline or calibrator; skipping.")
        return
    if preprocessor is None:
        print(f"  [WARN] no preprocessor in {rf_path}; skipping.")
        return

    try:
        feature_names = list(preprocessor.get_feature_names_out())
        if selector is not None:
            mask = selector.get_support()
            feature_names = [n for n, keep in zip(feature_names, mask) if keep]
    except Exception as exc:  # pragma: no cover - defensive
        print(f"  [WARN] could not extract feature names ({exc}); using indices.")
        feature_names = [f"f{i}" for i in range(len(importances))]

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_rf_feature_importance(
        feature_names, importances, ax=ax,
        top_n=10, dataset_label=DATASET_LABELS[dataset],
    )
    fig.tight_layout()
    out = FIGURES_DIR / f"{dataset}__rf_feature_importance.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out}")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if not ARTIFACTS_DIR.exists():
        print(
            f"[FAIL] {ARTIFACTS_DIR} does not exist. Run main.py first to "
            f"generate per-fold CV arrays."
        )
        return

    for dataset in DATASETS:
        print(f"\n=== {DATASET_LABELS[dataset]} ===")
        fold_arrays: Dict[str, list] = {}
        for model in MODELS:
            folds = _load_fold_list(dataset, model)
            if folds is not None:
                fold_arrays[model] = folds

        if not fold_arrays:
            print(f"  [SKIP] no CV arrays found for {dataset}.")
            continue

        _render_threshold_cost(dataset, fold_arrays)
        _render_pr_curves(dataset, fold_arrays)
        _render_roc_curves(dataset, fold_arrays)
        _render_calibration(dataset, fold_arrays)
        _render_dca(dataset, fold_arrays)
        _render_confusion(dataset, fold_arrays)
        _render_rf_feature_importance(dataset)

    print(f"\nFigures written to {FIGURES_DIR.resolve()}")


if __name__ == "__main__":
    main()
