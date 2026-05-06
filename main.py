import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

# Set seeds for reproducibility across torch and numpy architectures.
torch.manual_seed(67)
np.random.seed(67)

# Force UTF-8 stdout for cross-platform compatibility.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

from src.data_loader import load_alzheimers, load_autism, load_parkinsons_v3
from src.evaluation import evaluate_pipeline


# Display name → train_production.py production key. Keys match those used by
# train_production.py so thresholds.json can be consumed without translation.
PRODUCTION_KEY = {
    "Autism Screening": "Autism_Screening",
    "OASIS Alzheimer's": "OASIS_Alzheimers",
    "Parkinson's (Sakar)": "Parkinsons_Sakar",
}


def _print_summary(
    dataset_name: str,
    summary: Dict[str, Dict[str, float]],
    cv_label: str,
) -> None:
    """Format and print CV summary table for a single dataset."""
    print(f"\n{'─'*60}")
    print(f"  CV SUMMARY — {dataset_name} ({cv_label})")
    print(f"{'─'*60}")
    for model_name, scores in summary.items():
        print(
            f"  [{model_name:>3}] Acc: {scores['Accuracy']:.4f} ± {scores['Std_Accuracy']:.4f} | "
            f"F1: {scores['F1-Score']:.4f} ± {scores['Std_F1_Score']:.4f} | "
            f"Prec: {scores['Precision']:.4f} ± {scores['Std_Precision']:.4f} | "
            f"Rec: {scores['Recall']:.4f} ± {scores['Std_Recall']:.4f} | "
            f"AUC: {scores['AUC-ROC']:.4f} ± {scores['Std_AUC_ROC']:.4f}"
        )
        print(
            f"        Threshold(med): {scores['Threshold_Median']:.3f} ± {scores['Threshold_Std']:.3f} | "
            f"Sens: {scores['Sensitivity']:.4f} | "
            f"Spec: {scores['Specificity']:.4f} | "
            f"BalRec: {scores['Balanced_Recall']:.4f} | "
            f"Youden: {scores['Youden_J']:.4f} | "
            f"F2: {scores['F2_Score']:.4f}"
        )


def _collect_results(
    all_results: List[Dict[str, Any]],
    name: str,
    summary: Dict[str, Dict[str, float]],
) -> None:
    """Append summary metrics to the flat results list for CSV export."""
    for model, scores in summary.items():
        all_results.append(
            {
                "Dataset": name,
                "Model": model,
                "Mean_Accuracy": round(scores["Accuracy"], 4),
                "Std_Accuracy": round(scores["Std_Accuracy"], 4),
                "Mean_F1_Score": round(scores["F1-Score"], 4),
                "Std_F1_Score": round(scores["Std_F1_Score"], 4),
                "Mean_Precision": round(scores["Precision"], 4),
                "Std_Precision": round(scores["Std_Precision"], 4),
                "Mean_Recall": round(scores["Recall"], 4),
                "Std_Recall": round(scores["Std_Recall"], 4),
                "Mean_AUC_ROC": round(scores["AUC-ROC"], 4),
                "Std_AUC_ROC": round(scores["Std_AUC_ROC"], 4),
                "Threshold_Median": round(scores["Threshold_Median"], 4),
                "Threshold_Std": round(scores["Threshold_Std"], 4),
                "Sensitivity": round(scores["Sensitivity"], 4),
                "Specificity": round(scores["Specificity"], 4),
                "Balanced_Recall": round(scores["Balanced_Recall"], 4),
                "Std_Balanced_Recall": round(scores["Std_Balanced_Recall"], 4),
                "Youden_J": round(scores["Youden_J"], 4),
                "F2_Score": round(scores["F2_Score"], 4),
            }
        )


def _build_threshold_record(
    summary: Dict[str, Dict[str, float]],
    per_fold_thresholds: Dict[str, List[float]],
) -> Dict[str, Dict[str, Any]]:
    """Per-(dataset,model) record for thresholds.json — keyed by model name.

    Bundles the deployment threshold (median across folds) plus the CV-derived
    metrics so train_production.py can embed them into saved artifacts.
    """
    record: Dict[str, Dict[str, Any]] = {}
    for model_name, scores in summary.items():
        record[model_name] = {
            "threshold": float(scores["Threshold_Median"]),
            "threshold_std": float(scores["Threshold_Std"]),
            "per_fold_thresholds": [float(t) for t in per_fold_thresholds[model_name]],
            "cv_metrics": {
                "sensitivity": float(scores["Sensitivity"]),
                "specificity": float(scores["Specificity"]),
                "balanced_recall": float(scores["Balanced_Recall"]),
                "youden_j": float(scores["Youden_J"]),
                "f2": float(scores["F2_Score"]),
                "auc": float(scores["AUC-ROC"]),
                "accuracy": float(scores["Accuracy"]),
            },
        }
    return record


def main() -> None:
    """Execute the full ML evaluation pipeline across all datasets."""
    print("Initializing COMP3608 (5-Fold CV with recall_macro threshold sweep)...")

    all_results: List[Dict[str, Any]] = []
    thresholds_index: Dict[str, Dict[str, Any]] = {}

    # Datasets with one row per subject — safe for StratifiedKFold.
    # Autism labels are deterministic over A1–A10 (Class/ASD = 1 iff sum ≥ 6),
    # so threshold sweeping is uninformative; fall back to fixed thresholds.
    standard_datasets = {
        "Autism Screening": load_autism,
    }

    for name, loader in standard_datasets.items():
        try:
            print(f"\n{'='*60}")
            print(f"EVALUATING: {name}  (5-Fold Stratified CV, fixed thresholds)")
            print(f"{'='*60}")

            X, y = loader()
            print(f"  Loaded {name}. Shape: {X.shape}")

            summary, _, per_fold = evaluate_pipeline(
                X, y, run_threshold_sweep=False
            )
            _print_summary(name, summary, "Stratified")
            _collect_results(all_results, name, summary)
            thresholds_index[PRODUCTION_KEY[name]] = _build_threshold_record(
                summary, per_fold
            )
        except Exception as e:
            print(f"\n[ERROR] Failed on {name}: {e}")
            traceback.print_exc()

    # Datasets with multiple observations per subject — requires GroupKFold.
    grouped_datasets = {
        "OASIS Alzheimer's": load_alzheimers,
        "Parkinson's (Sakar)": load_parkinsons_v3,
    }

    for name, loader in grouped_datasets.items():
        try:
            print(f"\n{'='*60}")
            print(f"EVALUATING: {name}  (5-Fold GroupKFold CV, recall_macro sweep)")
            print(f"{'='*60}")

            X, y, groups = loader()
            n_subjects = groups.nunique()
            print(
                f"  Loaded {name}. Shape: {X.shape}, "
                f"Subjects: {n_subjects}, Observations/subject: ~{len(X) // n_subjects}"
            )

            summary, _, per_fold = evaluate_pipeline(
                X, y, groups=groups, run_threshold_sweep=True
            )
            _print_summary(name, summary, "GroupKFold")
            _collect_results(all_results, name, summary)
            thresholds_index[PRODUCTION_KEY[name]] = _build_threshold_record(
                summary, per_fold
            )
        except Exception as e:
            print(f"\n[ERROR] Failed on {name}: {e}")
            traceback.print_exc()

    # Save per-fold/median thresholds for train_production.py to consume.
    if thresholds_index:
        thresholds_path = Path("models") / "thresholds.json"
        thresholds_path.parent.mkdir(parents=True, exist_ok=True)
        with thresholds_path.open("w", encoding="utf-8") as f:
            json.dump(thresholds_index, f, indent=2, sort_keys=True)
        print(f"\n  Thresholds index written to {thresholds_path}")

    # Save results to CSV if any were generated.
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(
            "project_results_summary.csv", index=False, encoding="utf-8-sig"
        )
        print(f"\n{'='*60}")
        print(f"  Pipeline complete. Results saved to CSV.")
        print(f"{'='*60}\n")
        print(results_df.to_string(index=False))
    else:
        print("\n[WARNING] No results generated to save.")


if __name__ == "__main__":
    main()
