import copy
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch.utils.data import DataLoader, TensorDataset

from src.preprocessing import build_preprocessing_pipeline


def _default_hidden_dims(input_dim: int) -> List[int]:
    """Select hidden layer sizes based on the input feature dimensionality.

    Small clinical datasets (OASIS, ~8 features) get a compact 64→32 network
    matching literature recommendations, while high-dimensional datasets
    (Parkinson's TQWT, 753 features) use a wider 256→128→64 architecture.
    """
    if input_dim <= 20:
        return [64, 32]
    if input_dim <= 100:
        return [128, 64, 32]
    return [256, 128, 64]


class TabularFNN(nn.Module):
    """Fully connected neural network for tabular classification."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[List[int]] = None,
        dropout_rate: float = 0.3,
    ):
        """Initialize the network architecture.

        Args:
            input_dim: Number of input features.
            hidden_dims: Sizes of hidden layers (auto-selected from input_dim if None).
            dropout_rate: Dropout probability (default: 0.3).
        """
        super(TabularFNN, self).__init__()
        if hidden_dims is None:
            hidden_dims = _default_hidden_dims(input_dim)
        self.hidden_dims = hidden_dims

        layers: List[nn.Module] = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                ]
            )
            prev_dim = h_dim
        # Output logit — no sigmoid; handled by BCEWithLogitsLoss.
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass through the network.

        Args:
            x: Input tensor.

        Returns:
            Output logits (pre-sigmoid).
        """
        return self.network(x)


def train_evaluate_pytorch_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    epochs: int = 200,
    batch_size: int = 64,
    threshold: float = 0.5,
    hidden_dims: Optional[List[int]] = None,
    groups_train: Optional[pd.Series] = None,
    dropout_rate: Optional[float] = None,
    X_val_precarved: Optional[pd.DataFrame] = None,
    y_val_precarved: Optional[pd.Series] = None,
) -> Tuple[
    TabularFNN,
    ColumnTransformer,
    float,
    Dict[str, Any],
    np.ndarray,
    Optional[np.ndarray],
]:
    """Train and evaluate a PyTorch FNN model with early stopping.

    Owns its own preprocessing via build_preprocessing_pipeline — does not
    depend on any other model running first.

    When ``X_val_precarved`` and ``y_val_precarved`` are provided, the function
    skips its internal carve and uses the supplied slice for early stopping
    and threshold sweeping (the caller is responsible for guaranteeing
    group-disjointness and class balance). Otherwise the legacy 15-attempt
    GroupShuffleSplit retry path runs.

    Args:
        X_train: Raw training features (DataFrame with original dtypes).
        y_train: Training labels.
        X_test: Raw test features.
        y_test: Test labels.
        epochs: Maximum number of training epochs (default: 200).
        batch_size: Batch size for training (default: 64).
        threshold: Classification threshold for sigmoid output (default: 0.5).
        hidden_dims: Override for hidden layer widths.
        groups_train: Optional groups for the internal carve path.
        dropout_rate: Override for dropout probability.
        X_val_precarved: Pre-carved validation features (skip internal carve).
        y_val_precarved: Pre-carved validation labels (skip internal carve).

    Returns:
        Tuple (model, fitted_preprocessor, accuracy_at_threshold,
        classification_report_dict, test_positive_class_probs,
        val_positive_class_probs_or_None).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if X_val_precarved is not None and y_val_precarved is not None:
        # Caller carved the val slice; trust it and skip the retry block.
        X_train_sub = X_train
        y_train_sub = y_train
        X_val = X_val_precarved
        y_val = y_val_precarved
    elif groups_train is not None:
        # Legacy path: internal group-aware carve with retry.
        max_attempts = 15
        valid_split_found = False
        last_train_class_counts = None
        last_val_class_counts = None
        last_group_counts = None
        for attempt in range(max_attempts):
            gss = GroupShuffleSplit(
                n_splits=1, test_size=0.1, random_state=67 + attempt
            )
            train_idx_sub, val_idx = next(
                gss.split(X_train, y_train, groups=groups_train)
            )

            X_train_sub = X_train.iloc[train_idx_sub]
            X_val = X_train.iloc[val_idx]
            y_train_sub = y_train.iloc[train_idx_sub]
            y_val = y_train.iloc[val_idx]

            train_groups = set(groups_train.iloc[train_idx_sub])
            val_groups = set(groups_train.iloc[val_idx])
            if train_groups.intersection(val_groups):
                continue

            if len(set(y_val)) < 2:
                last_train_class_counts = np.bincount(
                    np.array(y_train_sub, dtype=int), minlength=2
                ).tolist()
                last_val_class_counts = np.bincount(
                    np.array(y_val, dtype=int), minlength=2
                ).tolist()
                last_group_counts = {
                    "train_groups": len(train_groups),
                    "val_groups": len(val_groups),
                }
                continue

            if len(set(y_train_sub)) < 2:
                last_train_class_counts = np.bincount(
                    np.array(y_train_sub, dtype=int), minlength=2
                ).tolist()
                last_val_class_counts = np.bincount(
                    np.array(y_val, dtype=int), minlength=2
                ).tolist()
                last_group_counts = {
                    "train_groups": len(train_groups),
                    "val_groups": len(val_groups),
                }
                continue

            valid_split_found = True
            break

        if not valid_split_found:
            raise ValueError(
                "Failed to find a class-valid, group-disjoint validation split "
                f"after {max_attempts} attempts. "
                f"Last seen train class counts={last_train_class_counts}, "
                f"val class counts={last_val_class_counts}, "
                f"group counts={last_group_counts}."
            )
    else:
        X_train_sub, X_val, y_train_sub, y_val = train_test_split(
            X_train, y_train, test_size=0.1, stratify=y_train, random_state=67
        )

    # FNN owns its own preprocessing — fit on training subset only.
    preprocessor = build_preprocessing_pipeline(X_train_sub)
    X_train_scaled = preprocessor.fit_transform(X_train_sub)
    X_val_scaled = preprocessor.transform(X_val)
    X_test_scaled = preprocessor.transform(X_test)

    # SMOTE: oversample minority class in training data only (never val/test).
    smote = SMOTE(random_state=67, k_neighbors=3)
    X_train_scaled, y_train_sub = smote.fit_resample(
        X_train_scaled, np.array(y_train_sub)
    )

    # DataLoader mini-batching.
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_scaled),
        torch.FloatTensor(np.array(y_train_sub)).unsqueeze(1),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    X_val_t = torch.FloatTensor(X_val_scaled).to(device)
    y_val_t = torch.FloatTensor(np.array(y_val)).unsqueeze(1).to(device)
    X_test_t = torch.FloatTensor(X_test_scaled).to(device)
    y_test_true = np.array(y_test)

    if dropout_rate is None:
        dropout_rate = 0.4 if X_train_scaled.shape[1] > 100 else 0.3

    model = TabularFNN(
        X_train_scaled.shape[1], hidden_dims=hidden_dims, dropout_rate=dropout_rate
    ).to(device)

    # Handle class imbalance via neg_to_pos_ratio (PyTorch docs formulation).
    neg_to_pos_ratio = float((np.array(y_train_sub) == 0).sum()) / max(
        float((np.array(y_train_sub) == 1).sum()), 1.0
    )
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([neg_to_pos_ratio], dtype=torch.float).to(device)
    )
    optimizer = optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-5
    )

    # Early stopping based on validation loss.
    best_val_loss = float("inf")
    best_weights = None
    patience = 20
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()

        # Evaluate on validation split.
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), y_val_t).item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_weights = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Restore best validation weights before final test evaluation.
    if best_weights:
        model.load_state_dict(best_weights)

    model.eval()
    with torch.no_grad():
        test_probs = torch.sigmoid(model(X_test_t)).cpu().numpy().flatten()
        val_probs = torch.sigmoid(model(X_val_t)).cpu().numpy().flatten()
        binary_preds = (test_probs >= threshold).astype(int)

    acc = accuracy_score(y_test_true, binary_preds)
    rep = cast(
        Dict[str, Any],
        classification_report(
            y_test_true, binary_preds, zero_division=0, output_dict=True
        ),
    )

    return (
        model,
        preprocessor,
        float(acc),
        rep,
        test_probs.astype(float),
        val_probs.astype(float),
    )


def train_pytorch_model(
    X: pd.DataFrame,
    y: pd.Series,
    epochs: int = 150,
    batch_size: int = 32,
    seed: int = 67,
    hidden_dims: Optional[List[int]] = None,
    dropout_rate: Optional[float] = None,
) -> Tuple[TabularFNN, ColumnTransformer]:
    """Train FNN on full dataset for production deployment.

    No train/val split or early stopping — trains for a fixed number of epochs.
    Use this for final model generation after CV evaluation has established
    expected performance.

    Args:
        X: Full feature matrix.
        y: Full target vector.
        epochs: Number of training epochs (default: 150).
        batch_size: Batch size (default: 32).
        seed: Random seed.

    Returns:
        Tuple of (trained_model, fitted_preprocessor).
    """
    torch.manual_seed(seed)

    preprocessor = build_preprocessing_pipeline(X)
    X_encoded = preprocessor.fit_transform(X)

    # SMOTE: oversample minority class before training.
    smote = SMOTE(random_state=seed, k_neighbors=3)
    X_encoded, y_resampled = smote.fit_resample(X_encoded, np.array(y))

    X_tensor = torch.tensor(X_encoded, dtype=torch.float32)
    y_tensor = torch.tensor(y_resampled, dtype=torch.float32).unsqueeze(1)

    if dropout_rate is None:
        dropout_rate = 0.4 if X_tensor.shape[1] > 100 else 0.3

    model = TabularFNN(
        input_dim=X_tensor.shape[1], hidden_dims=hidden_dims, dropout_rate=dropout_rate
    )

    # Calculate pos_weight for class imbalance.
    num_pos = y_tensor.sum()
    num_neg = len(y_tensor) - num_pos
    pos_weight = num_neg / num_pos if num_pos > 0 else torch.tensor(1.0)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    model.train()
    for epoch in range(epochs):
        permutation = torch.randperm(X_tensor.size(0))
        for i in range(0, X_tensor.size(0), batch_size):
            indices = permutation[i : i + batch_size]
            batch_x, batch_y = X_tensor[indices], y_tensor[indices]

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    return model, preprocessor
