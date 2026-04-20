import os
import re
import sys
import traceback
from typing import Any, Callable, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold

# Set seeds for reproducibility across torch and numpy architectures.
torch.manual_seed(67)
np.random.seed(67)

# Force UTF-8 stdout for cross-platform compatibility.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

from src.data_loader import load_alzheimers, load_autism, load_parkinsons_v2
from src.model_pytorch import train_evaluate_pytorch_model
from src.models_sklearn import (
    train_evaluate_logistic_regression,
    train_evaluate_random_forest,
)


def evaluate_pipeline_cv(
    dataset_name: str, X: pd.DataFrame, y: pd.Series, n_splits: int = 5
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, List[Dict[str, float]]]]:
    """Evaluate ML models using stratified k-fold cross-validation.

    Args:
        dataset_name: Name of the dataset for display and saving.
        X: Feature matrix.
        y: Target variable.
        n_splits: Number of cross-validation folds (default: 5).

    Returns:
        Tuple of (summary_dict, fold_metrics_dict) where:
            - summary_dict contains mean/std metrics for each model
            - fold_metrics_dict contains per-fold metrics
    """
    print(f"\n{'='*60}")
    print(f"EVALUATING: {dataset_name}  ({n_splits}-Fold Stratified CV)")
    print(f"{'='*60}")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=67)
    metrics = {"LR": [], "RF": [], "FNN": []}

    best_models = {
        "LR": {"model": None, "f1": -1},
        "RF": {"model": None, "f1": -1},
        "FNN": {"model": None, "f1": -1},
    }

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        # Enforce strict determinism locally per-fold
        torch.manual_seed(67 + fold)
        np.random.seed(67 + fold)

        print(f"\n--- Fold {fold}/{n_splits} ---")

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Note: Imputation is fully encapsulated within each model's pipeline.

        y_train = y_train.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)

        # Logistic Regression.
        lr_model, lr_acc, lr_rep = train_evaluate_logistic_regression(
            X_train, y_train, X_test, y_test
        )
        lr_f1 = float(lr_rep["macro avg"]["f1-score"])
        if lr_f1 > best_models["LR"]["f1"]:
            best_models["LR"] = {"model": lr_model, "f1": lr_f1}

        metrics["LR"].append(
            {
                "acc": lr_acc,
                "f1": lr_f1,
                "prec": lr_rep["macro avg"]["precision"],
                "rec": lr_rep["macro avg"]["recall"],
            }
        )

        # Random Forest.
        rf_model, rf_acc, rf_rep = train_evaluate_random_forest(
            X_train,
            y_train,
            X_test,
            y_test,
        )
        rf_f1 = float(rf_rep["macro avg"]["f1-score"])
        if rf_f1 > best_models["RF"]["f1"]:
            best_models["RF"] = {"model": rf_model, "f1": rf_f1}

        metrics["RF"].append(
            {
                "acc": rf_acc,
                "f1": rf_f1,
                "prec": rf_rep["macro avg"]["precision"],
                "rec": rf_rep["macro avg"]["recall"],
            }
        )

        # Reuse LR's fitted preprocessor to ensure FNN receives consistently encoded
        # and imputed data without fitting a separate transformer.
        fitted_preprocessor = lr_model.named_steps["preprocessor"]
        X_train_encoded = pd.DataFrame(fitted_preprocessor.transform(X_train))
        X_test_encoded = pd.DataFrame(fitted_preprocessor.transform(X_test))

        # Feed-Forward Neural Network.
        fnn_model, fnn_acc, fnn_rep = train_evaluate_pytorch_model(
            X_train_encoded, y_train, X_test_encoded, y_test
        )
        fnn_f1 = float(fnn_rep["macro avg"]["f1-score"])
        if fnn_f1 > best_models["FNN"]["f1"]:
            best_models["FNN"] = {"model": fnn_model, "f1": fnn_f1}

        metrics["FNN"].append(
            {
                "acc": fnn_acc,
                "f1": fnn_f1,
                "prec": fnn_rep["macro avg"]["precision"],
                "rec": fnn_rep["macro avg"]["recall"],
            }
        )

    # Save best model from each CV fold.
    # Note: Each fold only saw ~80% of data. For production, retrain on all data.
    # This snapshot is useful for initial model exploration.
    os.makedirs("models", exist_ok=True)
    safe_name = re.sub(r"[^\w]", "_", dataset_name)
    lr_path = f"models/{safe_name}_lr_pipeline.joblib"
    rf_path = f"models/{safe_name}_rf.joblib"
    fnn_path = f"models/{safe_name}_fnn_weights.pt"

    joblib.dump(best_models["LR"]["model"], lr_path)
    joblib.dump(best_models["RF"]["model"], rf_path)
    torch.save(best_models["FNN"]["model"].state_dict(), fnn_path)

    print(f"\n   [Saved BEST] LR  (F1: {best_models['LR']['f1']:.4f}) → {lr_path}")
    print(f"   [Saved BEST] RF  (F1: {best_models['RF']['f1']:.4f}) → {rf_path}")
    print(f"   [Saved BEST] FNN (F1: {best_models['FNN']['f1']:.4f}) → {fnn_path}")

    summary: Dict[str, Dict[str, float]] = {}
    print(f"\n{'─'*60}\n  CV SUMMARY — {dataset_name}\n{'─'*60}")
    for model_name, fold_metrics in metrics.items():
        mean_acc = float(np.mean([m["acc"] for m in fold_metrics]))
        mean_f1 = float(np.mean([m["f1"] for m in fold_metrics]))
        mean_prec = float(np.mean([m["prec"] for m in fold_metrics]))
        mean_rec = float(np.mean([m["rec"] for m in fold_metrics]))
        # Report ± std to expose fold-to-fold variance natively across all metrics.
        std_acc = float(np.std([m["acc"] for m in fold_metrics]))
        std_f1 = float(np.std([m["f1"] for m in fold_metrics]))
        std_prec = float(np.std([m["prec"] for m in fold_metrics]))
        std_rec = float(np.std([m["rec"] for m in fold_metrics]))

        summary[model_name] = {
            "Accuracy": mean_acc,
            "Std_Accuracy": std_acc,
            "F1-Score": mean_f1,
            "Std_F1_Score": std_f1,
            "Precision": mean_prec,
            "Std_Precision": std_prec,
            "Recall": mean_rec,
            "Std_Recall": std_rec,
        }
        print(
            f"  [{model_name:>3}] Acc: {mean_acc:.4f} ± {std_acc:.4f} | "
            f"F1: {mean_f1:.4f} ± {std_f1:.4f} | "
            f"Prec: {mean_prec:.4f} ± {std_prec:.4f} | "
            f"Rec: {mean_rec:.4f} ± {std_rec:.4f}"
        )

    return summary, metrics


def main() -> None:
    """Execute the full ML evaluation pipeline across all datasets."""
    print("Initializing COMP3608 (5-Fold Stratified CV)...")

    datasets: Dict[str, Callable[[], Tuple[pd.DataFrame, pd.Series]]] = {
        "OASIS Alzheimer's": load_alzheimers,
        "Parkinson's (Sakar)": load_parkinsons_v2,
        "Autism Screening": load_autism,
    }

    all_results: List[Dict[str, Any]] = []
    all_fold_metrics: Dict[str, Any] = {}
    for name, loader_func in datasets.items():
        try:
            X, y = loader_func()
            summary, fold_metrics = evaluate_pipeline_cv(name, X, y)
            all_fold_metrics[name] = fold_metrics
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
        except Exception as e:
            print(f"\n[ERROR] Failed on {name}: {e}")
            traceback.print_exc()

    # Save results to CSV if any were generated.
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(
            "project_results_summary.csv", index=False, encoding="utf-8-sig"
        )
        print(f"\n{'='*60}\n  Pipeline complete. Results saved to CSV.\n{'='*60}\n")
        print(results_df.to_string(index=False))
    else:
        print("\n[WARNING] No results generated to save.")


if __name__ == "__main__":
    main()
