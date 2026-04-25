import sys
import traceback
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
            f"Rec: {scores['Recall']:.4f} ± {scores['Std_Recall']:.4f}"
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
            }
        )


def main() -> None:
    """Execute the full ML evaluation pipeline across all datasets."""
    print("Initializing COMP3608 (5-Fold CV)...")

    all_results: List[Dict[str, Any]] = []

    # Datasets with one row per subject — safe for StratifiedKFold.
    standard_datasets = {
        "OASIS Alzheimer's": load_alzheimers,
        "Autism Screening": load_autism,
    }

    for name, loader in standard_datasets.items():
        try:
            print(f"\n{'='*60}")
            print(f"EVALUATING: {name}  (5-Fold Stratified CV)")
            print(f"{'='*60}")

            X, y = loader()
            print(f"  Loaded {name}. Shape: {X.shape}")

            summary, _ = evaluate_pipeline(X, y)
            _print_summary(name, summary, "Stratified")
            _collect_results(all_results, name, summary)
        except Exception as e:
            print(f"\n[ERROR] Failed on {name}: {e}")
            traceback.print_exc()

    # Parkinson's: multiple recordings per patient, requires GroupKFold.
    try:
        print(f"\n{'='*60}")
        print(f"EVALUATING: Parkinson's (Sakar)  (5-Fold GroupKFold CV)")
        print(f"{'='*60}")

        X_pk, y_pk, groups_pk = load_parkinsons_v3()
        n_patients = groups_pk.nunique()
        print(
            f"  Loaded Parkinson's (Sakar). Shape: {X_pk.shape}, "
            f"Patients: {n_patients}, Recordings/patient: ~{len(X_pk) // n_patients}"
        )

        pk_summary, _ = evaluate_pipeline(X_pk, y_pk, groups=groups_pk)
        _print_summary("Parkinson's (Sakar)", pk_summary, "GroupKFold")
        _collect_results(all_results, "Parkinson's (Sakar)", pk_summary)
    except Exception as e:
        print(f"\n[ERROR] Failed on Parkinson's (Sakar): {e}")
        traceback.print_exc()

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
