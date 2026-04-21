import os
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from typing import Dict, Callable, Tuple
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform

# Import loaders and preprocessing builder
from src.data_loader import load_alzheimers, load_autism, load_parkinsons_v2
from src.models_sklearn import build_preprocessing_pipeline
from src.model_pytorch import TabularFNN  

# Enforce reproducibility
torch.manual_seed(67)

def train_production_sklearn(X: pd.DataFrame, y: pd.Series, model_type: str, seed: int = 67) -> Pipeline:
    """Trains a final Scikit-Learn model on 100% of the data optimizing for Recall."""
    preprocessor = build_preprocessing_pipeline(X)
    
    if model_type == "LR":
        base_model = LogisticRegression(max_iter=10000, random_state=seed, class_weight="balanced")
        param_dist = {"clf__C": loguniform(1e-3, 1e1), "clf__solver": ["lbfgs", "saga"]}
    elif model_type == "RF":
        base_model = RandomForestClassifier(random_state=seed, n_jobs=1, class_weight="balanced")
        param_dist = {
            "clf__n_estimators": [100, 150, 200, 300],
            "clf__max_depth": [5, 7, 10, None],
            "clf__min_samples_leaf": [2, 3, 5],
        }
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    pipeline = Pipeline([("preprocessor", preprocessor), ("clf", base_model)])
    
    # Optimizing for recall_macro to match clinical triage goals
    search = RandomizedSearchCV(
        pipeline, param_dist, n_iter=15, cv=3, scoring="recall_macro", 
        random_state=seed, n_jobs=-1, refit=True
    )
    
    search.fit(X, y)
    return search.best_estimator_

def train_production_fnn(X: pd.DataFrame, y: pd.Series, fitted_preprocessor, model_save_path: str, seed: int = 67) -> None:
    """Trains the PyTorch FNN on 100% of the encoded data."""
    torch.manual_seed(seed)
    
    # Transform data using the fitted LR preprocessor
    X_encoded = pd.DataFrame(fitted_preprocessor.transform(X))
    
    # Convert to Tensors
    X_tensor = torch.tensor(X_encoded.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.reset_index(drop=True).values, dtype=torch.float32).unsqueeze(1)
    
    # Initialize Model 
    input_dim = X_tensor.shape[1]
    model = TabularFNN(input_dim=input_dim)
    
    # Calculate pos_weight for class imbalance
    num_pos = y_tensor.sum()
    num_neg = len(y_tensor) - num_pos
    pos_weight = num_neg / num_pos if num_pos > 0 else torch.tensor(1.0)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    model.train()
    epochs = 150
    batch_size = 32
    
    # Full dataset training loop
    for epoch in range(epochs):
        permutation = torch.randperm(X_tensor.size()[0])
        for i in range(0, X_tensor.size()[0], batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = X_tensor[indices], y_tensor[indices]
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
    # Save the weights
    torch.save(model.state_dict(), model_save_path)

def generate_production_artifacts() -> None:
    """Generates and saves models trained on 100% of the dataset for deployment."""
    os.makedirs("models/production", exist_ok=True)
    
    datasets: Dict[str, Callable[[], Tuple[pd.DataFrame, pd.Series]]] = {
        "OASIS_Alzheimers": load_alzheimers,
        "Parkinsons_Sakar": load_parkinsons_v2,
        "Autism_Screening": load_autism,
    }

    for name, loader in datasets.items():
        print(f"\nProcessing Production Models for {name}...")
        try:
            X, y = loader()
            
            # Train LR
            lr_prod = train_production_sklearn(X, y, "LR")
            joblib.dump(lr_prod, f"models/production/{name}_lr.joblib")
            print("  [✓] Logistic Regression saved.")
            
            # Train RF
            rf_prod = train_production_sklearn(X, y, "RF")
            joblib.dump(rf_prod, f"models/production/{name}_rf.joblib")
            print("  [✓] Random Forest saved.")
            
            # Train FNN
            preprocessor = lr_prod.named_steps["preprocessor"]
            fnn_path = f"models/production/{name}_fnn_weights.pt"
            train_production_fnn(X, y, preprocessor, fnn_path)
            print("  [✓] PyTorch FNN saved.")
            
        except Exception as e:
            print(f"  [X] Failed processing {name}: {str(e)}")

if __name__ == "__main__":
    generate_production_artifacts()