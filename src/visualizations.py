"""Visualization plotters for stakeholder + assessor figures.

Pure plotting layer — every function takes a matplotlib Axes and pre-computed
arrays. I/O lives in make_visuals.py so this module is unit-testable in
isolation and the CV pipeline never depends on matplotlib at runtime.
"""
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

# Cost defaults used by the threshold-vs-Z plot. C_FN >> C_FP encodes the
# clinical asymmetry stated in the report: a missed sick patient is far more
# costly than a false alarm.
DEFAULT_C_FN = 5.0
DEFAULT_C_FP = 1.0

MODEL_COLORS = {"LR": "#1f77b4", "RF": "#2ca02c", "FNN": "#d62728"}


def _cost_at_threshold(
    y_true: np.ndarray, y_score: np.ndarray, t: float, c_fn: float, c_fp: float
) -> float:
    preds = (y_score >= t).astype(int)
    fn = int(((y_true == 1) & (preds == 0)).sum())
    fp = int(((y_true == 0) & (preds == 1)).sum())
    return c_fn * fn + c_fp * fp


def plot_threshold_cost_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    t_star: float,
    ax: Axes,
    c_fn: float = DEFAULT_C_FN,
    c_fp: float = DEFAULT_C_FP,
    model_label: str = "model",
) -> None:
    """Plot the cost function Z(t) = C_FN * FN(t) + C_FP * FP(t) over t in [0,1].

    Marks the operating point ``t_star`` chosen by the per-fold sweep with a
    dashed vertical line and an annotation showing Z(t_star). The argmin of the
    curve is also marked, so the gap between "what the sweep picked" and "what
    the cost function would prefer" is visible.
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    costs = np.array(
        [_cost_at_threshold(y_true, y_score, t, c_fn, c_fp) for t in thresholds]
    )

    ax.plot(thresholds, costs, color=MODEL_COLORS.get(model_label, "#444"), lw=2)
    ax.fill_between(thresholds, 0, costs, alpha=0.1,
                    color=MODEL_COLORS.get(model_label, "#444"))

    z_star = _cost_at_threshold(y_true, y_score, t_star, c_fn, c_fp)
    ax.axvline(t_star, color="black", ls="--", lw=1.2,
               label=f"Sweep t*={t_star:.2f}\nZ={z_star:.0f}")

    argmin_t = float(thresholds[int(np.argmin(costs))])
    z_min = float(costs.min())
    ax.scatter(
        [argmin_t], [z_min], marker="o", s=60, edgecolor="black",
        facecolor="gold", zorder=5,
        label=f"argmin Z @ t={argmin_t:.2f}\nZ={z_min:.0f}",
    )

    ax.set_xlabel("Decision threshold t")
    ax.set_ylabel(f"Cost  Z = {c_fn:.0f}·FN + {c_fp:.0f}·FP")
    ax.set_title(f"{model_label}: cost vs. threshold")
    ax.set_xlim(0.0, 1.0)
    ax.legend(loc="upper center", fontsize=8, framealpha=0.85)
    ax.grid(alpha=0.3)


def plot_pr_curves(
    arrays_by_model: Dict[str, Tuple[np.ndarray, np.ndarray]],
    ax: Axes,
    dataset_label: str,
) -> None:
    """Overlay precision-recall curves for LR/RF/FNN on a shared axis.

    ``arrays_by_model`` maps model name -> (y_true_concat, y_score_concat) where
    folds are concatenated to summarise overall ranking quality. Reports AUPRC
    in the legend so models can be ordered.
    """
    for model_name, (y_true, y_score) in arrays_by_model.items():
        if len(np.unique(y_true)) < 2:
            continue
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        # PR curves want increasing recall; precision_recall_curve already gives
        # arrays sorted by descending threshold, so recall is monotone increasing.
        ap = auc(recall, precision)
        ax.plot(
            recall, precision,
            color=MODEL_COLORS.get(model_name, None),
            lw=2,
            label=f"{model_name} (AUPRC={ap:.3f})",
        )

    base_rate = float(np.mean(np.concatenate(
        [yt for (yt, _) in arrays_by_model.values()]
    )))
    ax.axhline(
        base_rate, color="grey", ls=":", lw=1.0,
        label=f"baseline (prevalence={base_rate:.2f})",
    )

    ax.set_xlabel("Recall (sensitivity)")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall — {dataset_label}")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.85)
    ax.grid(alpha=0.3)


def plot_roc_curves(
    arrays_by_model: Dict[str, Tuple[np.ndarray, np.ndarray]],
    ax: Axes,
    dataset_label: str,
) -> None:
    """Overlay ROC curves for LR/RF/FNN with per-model AUC in the legend."""
    for model_name, (y_true, y_score) in arrays_by_model.items():
        if len(np.unique(y_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_score)
        au = roc_auc_score(y_true, y_score)
        ax.plot(
            fpr, tpr,
            color=MODEL_COLORS.get(model_name, None),
            lw=2,
            label=f"{model_name} (AUC={au:.3f})",
        )

    ax.plot([0, 1], [0, 1], color="grey", ls=":", lw=1.0, label="chance")
    ax.set_xlabel("False positive rate (1 - specificity)")
    ax.set_ylabel("True positive rate (sensitivity)")
    ax.set_title(f"ROC — {dataset_label}")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.85)
    ax.grid(alpha=0.3)


def plot_calibration_curves(
    arrays_by_model: Dict[str, Tuple[np.ndarray, np.ndarray]],
    ax: Axes,
    dataset_label: str,
    n_bins: int = 10,
) -> None:
    """Overlay reliability diagrams. Diagonal = perfect calibration.

    Curves below the diagonal mean the model is over-confident (predicted
    probabilities exceed the empirical positive rate); curves above mean
    under-confident. Useful for spotting when an FNN's logits drift across
    folds — see report's "calibrate the FNN" item.
    """
    for model_name, (y_true, y_score) in arrays_by_model.items():
        if len(np.unique(y_true)) < 2:
            continue
        # strategy="quantile" handles the imbalanced datasets better than
        # the default uniform binning (which can leave empty bins at the tails).
        prob_true, prob_pred = calibration_curve(
            y_true, y_score, n_bins=n_bins, strategy="quantile"
        )
        ax.plot(
            prob_pred, prob_true,
            color=MODEL_COLORS.get(model_name, None),
            marker="o", lw=2, markersize=5,
            label=model_name,
        )

    ax.plot([0, 1], [0, 1], color="grey", ls=":", lw=1.0, label="perfect")
    ax.set_xlabel("Mean predicted probability (per bin)")
    ax.set_ylabel("Observed positive fraction (per bin)")
    ax.set_title(f"Calibration — {dataset_label}")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.85)
    ax.grid(alpha=0.3)


def _net_benefit(y_true: np.ndarray, y_score: np.ndarray, p_t: float) -> float:
    """Net benefit at decision threshold p_t (the standard DCA quantity).

    NB(p_t) = TP/N − (FP/N) × (p_t / (1 − p_t))

    The (p_t / (1 − p_t)) factor encodes the clinical preference: at threshold
    p_t the user is implicitly weighting C_FP/C_FN = p_t/(1 − p_t). Sweeping p_t
    therefore generalises the cost-sensitive Z curve over *all* reasonable cost
    ratios in one plot — which is the audit's #1 priority over a fixed-ratio Z.
    """
    if p_t >= 0.999 or p_t <= 0.0:
        return 0.0
    n = len(y_true)
    pred = (y_score >= p_t).astype(int)
    tp = int(((y_true == 1) & (pred == 1)).sum())
    fp = int(((y_true == 0) & (pred == 1)).sum())
    return tp / n - (fp / n) * (p_t / (1.0 - p_t))


def _net_benefit_treat_all(y_true: np.ndarray, p_t: float) -> float:
    """Net benefit of the 'treat everyone' default strategy at threshold p_t."""
    if p_t >= 0.999 or p_t <= 0.0:
        return 0.0
    n = len(y_true)
    pos = int(y_true.sum())
    neg = n - pos
    return pos / n - (neg / n) * (p_t / (1.0 - p_t))


def plot_dca_curves(
    arrays_by_model: Dict[str, Tuple[np.ndarray, np.ndarray]],
    ax: Axes,
    dataset_label: str,
    p_t_max: float = 0.6,
    n_points: int = 60,
) -> None:
    """Decision curve analysis: net benefit vs threshold probability.

    For each model, plots NB(p_t) over a clinically relevant range of p_t.
    Overlays the two default strategies the audit explicitly cites:
      • treat all  — NB depends on prevalence and p_t
      • treat none — NB ≡ 0

    A model is *clinically useful* at a given p_t when its curve sits ABOVE
    both default lines — translation: using the model would correctly catch
    more sick patients than blanket-treating without proportionally more
    false referrals. The wider that range of p_t is, the more robust the
    model is to a clinician's actual cost preference.
    """
    p_t_grid = np.linspace(0.01, p_t_max, n_points)

    # Treat-all baseline uses the union of folds; identical for any model.
    any_y_true = next(iter(arrays_by_model.values()))[0]
    nb_all = np.array([_net_benefit_treat_all(any_y_true, p) for p in p_t_grid])

    for model_name, (y_true, y_score) in arrays_by_model.items():
        if len(np.unique(y_true)) < 2:
            continue
        nb = np.array([_net_benefit(y_true, y_score, p) for p in p_t_grid])
        ax.plot(
            p_t_grid, nb,
            color=MODEL_COLORS.get(model_name, None),
            lw=2,
            label=model_name,
        )

    ax.plot(p_t_grid, nb_all, color="#888", ls="--", lw=1.4, label="treat all")
    ax.axhline(0.0, color="#444", ls=":", lw=1.0, label="treat none")

    ax.set_xlabel("Threshold probability  p_t  (≈ implicit C_FP / (C_FN+C_FP))")
    ax.set_ylabel("Net benefit")
    ax.set_title(f"Decision curve analysis — {dataset_label}")
    ax.set_xlim(0.0, p_t_max)
    # Anchor a sensible y range; clip extreme negatives at low p_t.
    nb_min = float(min(nb_all.min(), -0.05))
    ax.set_ylim(max(nb_min, -0.2), max(0.5, nb_all.max() + 0.05))
    ax.legend(loc="upper right", fontsize=9, framealpha=0.85)
    ax.grid(alpha=0.3)


def plot_cost_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ax: Axes,
    model_label: str,
    class_names: Tuple[str, str] = ("Healthy", "At-risk"),
) -> None:
    """Render a 2x2 confusion matrix with the False-Negative cell highlighted.

    The FN annotation is the headline number for stakeholders — it answers
    "how many sick patients did we miss?" directly.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    total_positives = int(tp + fn)
    fn_rate = (fn / total_positives) if total_positives > 0 else 0.0

    # Highlight the FN cell by overlaying a coloured patch underneath the
    # default heatmap. We render the heatmap by hand so the FN cell stays red
    # regardless of count magnitude.
    ax.imshow(
        np.ones_like(cm, dtype=float),
        cmap="Blues", vmin=0, vmax=1, aspect="auto",
    )
    fn_overlay = np.zeros_like(cm, dtype=float)
    fn_overlay[1, 0] = 1.0
    ax.imshow(fn_overlay, cmap="Reds", vmin=0, vmax=1, alpha=0.55, aspect="auto")

    cell_text = [
        [f"TN\n{tn}", f"FP\n{fp}"],
        [f"FN\n{fn}", f"TP\n{tp}"],
    ]
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, cell_text[i][j],
                ha="center", va="center",
                fontsize=14, fontweight="bold",
                color=("black" if (i, j) != (1, 0) else "white"),
            )

    ax.set_xticks([0, 1], class_names)
    ax.set_yticks([0, 1], class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(
        f"{model_label}: confusion matrix\n"
        f"FN rate = {fn_rate:.1%} of true positives "
        f"(missed sick patients)",
        fontsize=10,
    )


def plot_rf_feature_importance(
    feature_names: List[str],
    importances: np.ndarray,
    ax: Axes,
    top_n: int = 10,
    dataset_label: str = "",
) -> None:
    """Horizontal bar chart of the top-N RF feature importances.

    Caller is responsible for extracting names + importances from the fitted
    pipeline (see make_visuals.py); keeping I/O out of the plotter avoids
    coupling matplotlib to the persistence layer.
    """
    importances = np.asarray(importances, dtype=float)
    if importances.size == 0:
        ax.set_title(f"{dataset_label}: no feature importances available")
        return

    order = np.argsort(importances)[::-1][:top_n]
    top_imp = importances[order]
    top_names = [feature_names[i] for i in order]

    ypos = np.arange(len(order))[::-1]
    ax.barh(ypos, top_imp, color="#2ca02c", edgecolor="black")
    ax.set_yticks(ypos, top_names)
    ax.set_xlabel("Importance (Gini)")
    ax.set_title(f"Top {len(order)} predictive features — {dataset_label}")
    ax.grid(axis="x", alpha=0.3)


def concat_folds(
    fold_list: List[Dict[str, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Concatenate per-fold arrays and return (y_true, y_score, median_threshold).

    Concatenation is the standard way to draw a single PR/threshold curve from
    k-fold output: each test fold is held-out, so the union covers each row
    exactly once (StratifiedKFold) or once per group (GroupKFold).
    """
    y_true = np.concatenate([f["y_true"] for f in fold_list])
    y_score = np.concatenate([f["y_score"] for f in fold_list])
    thresholds = np.concatenate([f["threshold"] for f in fold_list])
    return y_true, y_score, float(np.median(thresholds))
