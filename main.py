import os
import sys

# Force UTF-8 stdout
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer

from src.data_loader import load_alzheimers, load_parkinsons_v2, load_autism
from src.model_pytorch import train_evaluate_pytorch_model
# TODO: @Levi import your sklearn models here tomorrow morning or tonight idk, also you forgot to create your branches for em

def evaluate_pipeline_cv(dataset_name, X, y, n_splits=5):
    print(f"\n{'='*60}")
    print(f"EVALUATING: {dataset_name}  ({n_splits}-Fold Stratified CV)")
    print(f"{'='*60}")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    metrics = {'FNN': []} # TODO: @Levi add 'LR' and 'RF' lists here

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n--- Fold {fold}/{n_splits} ---")

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        imputer = SimpleImputer(strategy='median')
        X_train_clean = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
        X_test_clean = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

        y_train = y_train.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)

        # TODO: @Levi call train_evaluate_logistic_regression and train_evaluate_random_forest here

        # Feed-Forward Neural Network
        _, fnn_acc, fnn_rep = train_evaluate_pytorch_model(
            X_train_clean, y_train, X_test_clean, y_test
        )
        metrics['FNN'].append({
            'acc': fnn_acc,
            'f1':  fnn_rep['macro avg']['f1-score'],
            'prec': fnn_rep['macro avg']['precision'],
            'rec':  fnn_rep['macro avg']['recall']
        })

    summary = {}
    print(f"\n{'─'*60}\n  CV SUMMARY — {dataset_name}\n{'─'*60}")
    for model_name, fold_metrics in metrics.items():
        mean_acc  = np.mean([m['acc']  for m in fold_metrics])
        mean_f1   = np.mean([m['f1']   for m in fold_metrics])
        mean_prec = np.mean([m['prec'] for m in fold_metrics])
        mean_rec  = np.mean([m['rec']  for m in fold_metrics])
        summary[model_name] = {'Accuracy': mean_acc, 'F1-Score': mean_f1, 'Precision': mean_prec, 'Recall': mean_rec}
        print(f"  [{model_name:>3}] Acc: {mean_acc:.4f} | F1: {mean_f1:.4f} | Prec: {mean_prec:.4f} | Rec: {mean_rec:.4f}")

    return summary

def main():
    print("Initializing COMP3608 (5-Fold Stratified CV)...")

    datasets = {
        "OASIS Alzheimer's": load_alzheimers,
        "Parkinson's (Sakar)": load_parkinsons_v2,
        "Autism Screening": load_autism
    }

    all_results = []
    for name, loader_func in datasets.items():
        try:
            X, y = loader_func()
            summary = evaluate_pipeline_cv(name, X, y)
            for model, scores in summary.items():
                all_results.append({
                    'Dataset': name, 'Model': model,
                    'Mean_Accuracy': round(scores['Accuracy'], 4), 'Mean_F1_Score': round(scores['F1-Score'], 4),
                    'Mean_Precision': round(scores['Precision'], 4), 'Mean_Recall': round(scores['Recall'], 4)
                })
        except Exception as e:
            print(f"\n[ERROR] Failed on {name}: {e}")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv("project_results_summary.csv", index=False, encoding='utf-8-sig')
    print(f"\n{'='*60}\n  Pipeline complete. Results saved to CSV.\n{'='*60}\n")
    print(results_df.to_string(index=False))

if __name__ == "__main__":
    main()