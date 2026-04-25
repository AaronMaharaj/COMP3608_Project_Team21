import sys

import torch
from sklearn.model_selection import GroupKFold

from src.data_loader import load_alzheimers, load_autism, load_parkinsons_v3
from src.model_pytorch import train_pytorch_model
from src.persistence import save_fnn_model, save_sklearn_model
from src.pipeline_factory import build_lr_search, build_rf_search

# Enforce reproducibility.
torch.manual_seed(67)

# Force UTF-8 stdout for cross-platform compatibility.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]


def generate_production_artifacts() -> None:
    """Generate and save models trained on 100% of the dataset for deployment.

    Sklearn models use RandomizedSearchCV with inner CV for hyperparameter
    tuning on the full dataset. FNN trains for a fixed number of epochs.
    All artifacts are saved to models/production/.
    """
    datasets = {
        "OASIS_Alzheimers": {"loader": load_alzheimers, "grouped": False},
        "Autism_Screening": {"loader": load_autism, "grouped": False},
        "Parkinsons_Sakar": {"loader": load_parkinsons_v3, "grouped": True},
    }

    for name, config in datasets.items():
        print(f"\nProcessing Production Models for {name}...")
        try:
            if config["grouped"]:
                X, y, groups = config["loader"]()
            else:
                X, y = config["loader"]()
                groups = None

            # Determine inner CV strategy: group-aware for grouped datasets.
            inner_cv = GroupKFold(n_splits=3) if groups is not None else 3

            # Train and save LR.
            lr_search = build_lr_search(X, cv=inner_cv, n_iter=15)
            lr_search.fit(X, y, groups=groups)
            save_sklearn_model(
                lr_search.best_estimator_,
                f"models/production/{name}_lr.joblib",
            )
            print("  [✓] Logistic Regression saved.")

            # Train and save RF.
            rf_search = build_rf_search(X, cv=inner_cv, n_iter=15)
            rf_search.fit(X, y, groups=groups)
            save_sklearn_model(
                rf_search.best_estimator_,
                f"models/production/{name}_rf.joblib",
            )
            print("  [✓] Random Forest saved.")

            # Train and save FNN (bundles preprocessor with weights).
            fnn_model, fnn_preprocessor = train_pytorch_model(X, y)
            save_fnn_model(
                fnn_model,
                fnn_preprocessor,
                f"models/production/{name}_fnn.joblib",
            )
            print("  [✓] PyTorch FNN saved.")

        except Exception as e:
            print(f"  [X] Failed processing {name}: {e}")


if __name__ == "__main__":
    generate_production_artifacts()