from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import torch

from src.model_pytorch import TabularFNN


def save_sklearn_model(model: Any, path: str) -> None:
    """Save a fitted sklearn Pipeline via joblib.

    Args:
        model: Fitted sklearn Pipeline or estimator.
        path: Destination file path.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def save_fnn_model(
    model: TabularFNN,
    preprocessor: Any,
    path: str,
) -> None:
    """Save FNN weights + fitted preprocessor as a single joblib artifact.

    The artifact is a dict containing state_dict, input_dim, and preprocessor
    so that inference can reconstruct the full pipeline without external state.

    Args:
        model: Trained TabularFNN instance.
        preprocessor: Fitted ColumnTransformer used to preprocess raw inputs.
        path: Destination file path (use .joblib extension).
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    artifact: Dict[str, Any] = {
        "state_dict": model.state_dict(),
        "input_dim": model.network[0].in_features,
        "preprocessor": preprocessor,
    }
    joblib.dump(artifact, path)


def load_fnn_model(path: str) -> Tuple[TabularFNN, Any]:
    """Load a bundled FNN artifact.

    Args:
        path: Path to the saved .joblib artifact.

    Returns:
        Tuple of (model, fitted_preprocessor).
    """
    artifact = joblib.load(path)
    model = TabularFNN(input_dim=artifact["input_dim"])
    model.load_state_dict(artifact["state_dict"])
    model.eval()
    return model, artifact["preprocessor"]
