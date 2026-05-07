import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

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


# Default thresholds when no thresholds.json is present (e.g. running this
# script standalone before main.py has been executed). 0.35 for the FNN
# matches the historical hardcoded value; 0.5 for sklearn matches predict().
DEFAULT_THRESHOLDS: Dict[str, float] = {"LR": 0.5, "RF": 0.5, "FNN": 0.35}


def _load_thresholds_index(
    path: Path = Path("models") / "thresholds.json",
) -> Dict[str, Dict[str, Any]]:
    """Load thresholds.json produced by main.py, or return empty dict if absent."""
    if not path.exists():
        print(
            f"  [INFO] {path} not found. Falling back to default thresholds "
            f"({DEFAULT_THRESHOLDS}). Run main.py first to generate it."
        )
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve(
    thresholds_index: Dict[str, Dict[str, Any]], dataset_key: str, model_name: str
) -> tuple[float, Optional[Dict[str, float]]]:
    """Resolve (threshold, cv_metrics) for a given (dataset, model) pair."""
    record = thresholds_index.get(dataset_key, {}).get(model_name)
    if record is None:
        return DEFAULT_THRESHOLDS[model_name], None
    return float(record["threshold"]), record.get("cv_metrics")


def generate_production_artifacts() -> None:
    """Generate and save models trained on 100% of the dataset for deployment.

    Sklearn models use RandomizedSearchCV with inner CV for hyperparameter
    tuning on the full dataset. FNN trains for a fixed number of epochs.
    All artifacts are saved to models/production/ with the operating threshold
    and CV-derived metrics bundled in.
    """
    thresholds_index = _load_thresholds_index()

    datasets = {
        "OASIS_Alzheimers": {"loader": load_alzheimers, "grouped": True},
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
            lr_t, lr_cv = _resolve(thresholds_index, name, "LR")
            lr_search = build_lr_search(X, cv=inner_cv, n_iter=15)
            lr_search.fit(X, y, groups=groups)
            save_sklearn_model(
                lr_search.best_estimator_,
                f"models/production/{name}_lr.joblib",
                threshold=lr_t,
                cv_metrics=lr_cv,
            )
            print(f"  [OK] Logistic Regression saved (threshold={lr_t:.3f}).")

            # Train and save RF.
            rf_t, rf_cv = _resolve(thresholds_index, name, "RF")
            rf_search = build_rf_search(X, cv=inner_cv, n_iter=15)
            rf_search.fit(X, y, groups=groups)
            save_sklearn_model(
                rf_search.best_estimator_,
                f"models/production/{name}_rf.joblib",
                threshold=rf_t,
                cv_metrics=rf_cv,
            )
            print(f"  [OK] Random Forest saved (threshold={rf_t:.3f}).")

            # Train and save FNN (bundles preprocessor with weights).
            fnn_t, fnn_cv = _resolve(thresholds_index, name, "FNN")
            fnn_model, fnn_preprocessor = train_pytorch_model(X, y)
            save_fnn_model(
                fnn_model,
                fnn_preprocessor,
                f"models/production/{name}_fnn.joblib",
                threshold=fnn_t,
                cv_metrics=fnn_cv,
            )
            print(f"  [OK] PyTorch FNN saved (threshold={fnn_t:.3f}).")

        except Exception as e:
            print(f"  [FAIL] processing {name}: {e}")


if __name__ == "__main__":
    generate_production_artifacts()
