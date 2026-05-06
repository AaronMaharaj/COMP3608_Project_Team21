from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import torch

from src.model_pytorch import TabularFNN

ARTIFACT_VERSION = 2


def save_sklearn_model(
    model: Any,
    path: str,
    threshold: float = 0.5,
    cv_metrics: Optional[Dict[str, float]] = None,
) -> None:
    """Save a fitted sklearn Pipeline as a versioned dict via joblib.

    Bundles the operating threshold and any CV-derived metrics (sensitivity,
    specificity, etc.) alongside the pipeline so downstream consumers
    (e.g. Streamlit UIs) can apply the correct decision boundary without
    re-deriving it.

    Args:
        model: Fitted sklearn Pipeline or estimator.
        path: Destination file path.
        threshold: Decision threshold for predict_proba >= threshold (default: 0.5).
        cv_metrics: Optional dict of CV-derived metrics to embed (sensitivity,
            specificity, balanced_recall, youden_j, auc, etc.).
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    artifact: Dict[str, Any] = {
        "version": ARTIFACT_VERSION,
        "kind": "sklearn",
        "model": model,
        "threshold": float(threshold),
        "cv_metrics": cv_metrics or {},
    }
    joblib.dump(artifact, path)


def load_sklearn_model(path: str) -> Tuple[Any, float, Dict[str, float]]:
    """Load a sklearn artifact, handling both v2 (dict) and legacy (raw) formats.

    Args:
        path: Path to the saved .joblib artifact.

    Returns:
        Tuple (pipeline, threshold, cv_metrics). For legacy artifacts the
        threshold defaults to 0.5 and cv_metrics is empty.
    """
    loaded = joblib.load(path)
    if isinstance(loaded, dict) and loaded.get("kind") == "sklearn":
        return loaded["model"], float(loaded.get("threshold", 0.5)), loaded.get(
            "cv_metrics", {}
        )
    # Legacy: raw pipeline with no threshold metadata.
    return loaded, 0.5, {}


def save_fnn_model(
    model: TabularFNN,
    preprocessor: Any,
    path: str,
    threshold: float = 0.5,
    cv_metrics: Optional[Dict[str, float]] = None,
) -> None:
    """Save FNN weights + fitted preprocessor as a single joblib artifact.

    The artifact is a dict containing state_dict, input_dim, preprocessor,
    threshold, and CV-derived metrics so that inference can reconstruct the
    full pipeline without external state.

    Args:
        model: Trained TabularFNN instance.
        preprocessor: Fitted ColumnTransformer used to preprocess raw inputs.
        path: Destination file path (use .joblib extension).
        threshold: Decision threshold applied to sigmoid(logits) >= threshold.
        cv_metrics: Optional CV-derived metrics dict.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    artifact: Dict[str, Any] = {
        "version": ARTIFACT_VERSION,
        "kind": "fnn",
        "state_dict": model.state_dict(),
        "input_dim": model.network[0].in_features,
        "hidden_dims": model.hidden_dims,
        "preprocessor": preprocessor,
        "threshold": float(threshold),
        "cv_metrics": cv_metrics or {},
    }
    joblib.dump(artifact, path)


def load_fnn_model(path: str) -> Tuple[TabularFNN, Any]:
    """Load a bundled FNN artifact (legacy two-tuple return).

    Kept for backwards compatibility with any existing consumers; new callers
    should prefer ``load_fnn_artifact`` to also retrieve the threshold.

    Args:
        path: Path to the saved .joblib artifact.

    Returns:
        Tuple of (model, fitted_preprocessor).
    """
    artifact = joblib.load(path)
    model = TabularFNN(
        input_dim=artifact["input_dim"],
        hidden_dims=artifact.get("hidden_dims"),
    )
    model.load_state_dict(artifact["state_dict"])
    model.eval()
    return model, artifact["preprocessor"]


def load_fnn_artifact(
    path: str,
) -> Tuple[TabularFNN, Any, float, Dict[str, float]]:
    """Load a bundled FNN artifact including threshold and CV metrics.

    Args:
        path: Path to the saved .joblib artifact.

    Returns:
        Tuple (model, fitted_preprocessor, threshold, cv_metrics). Threshold
        defaults to 0.5 when missing (legacy artifacts).
    """
    artifact = joblib.load(path)
    model = TabularFNN(
        input_dim=artifact["input_dim"],
        hidden_dims=artifact.get("hidden_dims"),
    )
    model.load_state_dict(artifact["state_dict"])
    model.eval()
    return (
        model,
        artifact["preprocessor"],
        float(artifact.get("threshold", 0.5)),
        artifact.get("cv_metrics", {}),
    )
