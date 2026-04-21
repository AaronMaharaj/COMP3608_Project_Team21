import copy
from typing import Any, Dict, cast

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class TabularFNN(nn.Module):
    """Fully connected neural network for tabular classification."""

    def __init__(self, input_dim, dropout_rate=0.3):
        """Initialize the network architecture.

        Args:
            input_dim: Number of input features.
            dropout_rate: Dropout probability (default: 0.3).
        """
        super(TabularFNN, self).__init__()
        # BatchNorm helps convergence on collinear features (especially Parkinson's TQWT).
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 1),
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
    X_train, y_train, X_test, y_test, epochs=200, batch_size=64, threshold=0.5
):
    """Train and evaluate a PyTorch FNN model with early stopping.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_test: Test features.
        y_test: Test labels.
        epochs: Maximum number of training epochs (default: 200).
        batch_size: Batch size for training (default: 64).

    Returns:
        Tuple of (model, accuracy, classification_report_dict).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Validation split for pure early stopping (prevent memorization).
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(
        X_train, y_train, test_size=0.1, stratify=y_train
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_sub)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

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

    # Early stopping based on validation loss.
    best_val_loss = float("inf")
    best_weights = None
    patience = 15
    patience_counter = 0
    epoch_stopped = epochs

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

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_weights = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= patience:
                epoch_stopped = epoch + 1
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

    macro_avg = cast(Dict[str, float], rep.get("macro avg", {}))
    f1_score = macro_avg.get("f1-score", 0.0)
    precision = macro_avg.get("precision", 0.0)
    recall = macro_avg.get("recall", 0.0)

    print(
        f"   [FNN] Epochs: {epoch_stopped:03d} | Acc: {acc:.4f} | "
        f"F1: {f1_score:.4f} | "
        f"Prec: {precision:.4f} | Rec: {recall:.4f}"
    )
    return model, acc, rep
