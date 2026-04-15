import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

def train_evaluate_logistic_regression(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr_model = LogisticRegression(
        C=0.1,
        max_iter=2000,
        random_state=42,
        solver='lbfgs',
        class_weight='balanced'
    )
    lr_model.fit(X_train_scaled, y_train)
    predictions = lr_model.predict(X_test_scaled)

    acc = accuracy_score(y_test, predictions)
    rep = classification_report(y_test, predictions, zero_division=0, output_dict=True)

    print(f"   [LR]  Acc: {acc:.4f} | F1: {rep['macro avg']['f1-score']:.4f} | "
          f"Prec: {rep['macro avg']['precision']:.4f} | Rec: {rep['macro avg']['recall']:.4f}")
    return lr_model, acc, rep

def train_evaluate_random_forest(X_train, y_train, X_test, y_test, print_features=False):
    rf_model = RandomForestClassifier(
        n_estimators=150,
        max_depth=7,          # Constrained to prevent 100% train memorisation
        min_samples_leaf=3,   # Forces generalisation across samples
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    rf_model.fit(X_train, y_train)

    train_preds = rf_model.predict(X_train)
    test_preds = rf_model.predict(X_test)

    train_acc = accuracy_score(y_train, train_preds)
    test_acc = accuracy_score(y_test, test_preds)
    rep = classification_report(y_test, test_preds, zero_division=0, output_dict=True)

    print(f"   [RF]  Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f} | "
          f"F1: {rep['macro avg']['f1-score']:.4f} | "
          f"Prec: {rep['macro avg']['precision']:.4f} | Rec: {rep['macro avg']['recall']:.4f}")

    # Only print feature importances on the final fold to avoid terminal clutter
    if print_features and isinstance(X_train, pd.DataFrame):
        importances = rf_model.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False).head(5)
        print("\n   [Insight] Top 5 Features (Final Fold):")
        print(importance_df.to_string(index=False))

    return rf_model, test_acc, rep