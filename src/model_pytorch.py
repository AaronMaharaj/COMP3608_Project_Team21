import copy
from typing import Any, Dict, Tuple, cast

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from src.preprocessing import build_preprocessing_pipeline


class TabularFNN(nn.Module):
    """Fully connected neural network for tabular classification."""

    def __init__(self, input_dim, dropout_rate=0.3):
        """Initialize the network architecture.

        Args:
            input_dim: Number of input features.
            dropout_rate: Dropout probability (default: 0.3).
        """
        super(TabularFNN, self).__init__()
        # Wider layers reduce the compression bottleneck on high-dimensional inputs
        # (e.g. 753 Parkinson's TQWT features). BatchNorm stabilizes collinear inputs.
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1),
            # No sigmoid here—handled inside BCEWithLogitsLoss for numerical stability.
        )

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
) -> Tuple[TabularFNN, ColumnTransformer, float, Dict[str, Any]]:
    """Train and evaluate a PyTorch FNN model with early stopping.

    Owns its own preprocessing via build_preprocessing_pipeline — does not
    depend on any other model running first.

    Args:
        X_train: Raw training features (DataFrame with original dtypes).
        y_train: Training labels.
        X_test: Raw test features.
        y_test: Test labels.
        epochs: Maximum number of training epochs (default: 200).
        batch_size: Batch size for training (default: 64).
        threshold: Classification threshold for sigmoid output (default: 0.5).

    Returns:
        Tuple of (model, fitted_preprocessor, accuracy, classification_report_dict).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Validation split for early stopping (prevent memorization).
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(
        X_train, y_train, test_size=0.1, stratify=y_train
    )

    # FNN owns its own preprocessing — fit on training subset only.
    preprocessor = build_preprocessing_pipeline(X_train_sub)
    X_train_scaled = preprocessor.fit_transform(X_train_sub)
    X_val_scaled = preprocessor.transform(X_val)
    X_test_scaled = preprocessor.transform(X_test)

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

    model = TabularFNN(X_train_scaled.shape[1]).to(device)

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
        probs = torch.sigmoid(model(X_test_t))
        binary_preds = (probs >= threshold).cpu().int().numpy().flatten()

    acc = accuracy_score(y_test_true, binary_preds)
    rep = cast(
        Dict[str, Any],
        classification_report(
            y_test_true, binary_preds, zero_division=0, output_dict=True
        ),
    )

    return model, preprocessor, float(acc), rep


def train_pytorch_model(
    X: pd.DataFrame,
    y: pd.Series,
    epochs: int = 150,
    batch_size: int = 32,
    seed: int = 67,
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

    X_tensor = torch.tensor(X_encoded, dtype=torch.float32)
    y_tensor = torch.tensor(
        y.reset_index(drop=True).values, dtype=torch.float32
    ).unsqueeze(1)

    model = TabularFNN(input_dim=X_tensor.shape[1])

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
