import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report

# for exponential C value search (1e-3 to 10) instead of linear
from scipy.stats import loguniform


def train_evaluate_logistic_regression(X_train, y_train, X_test, y_test, seed=42):
    
    # n_iter=10 keeps the search fast enough for the outer 5-fold loop.
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=2000, random_state=seed, class_weight='balanced'))
    ])
    param_dist = {
        'clf__C':      loguniform(1e-3, 1e1),
        'clf__solver': ['lbfgs', 'saga'],
    }
    search = RandomizedSearchCV(
        pipeline, param_dist,
        n_iter=10, cv=3, scoring='f1_macro',
        random_state=seed, n_jobs=-1
    )
    search.fit(X_train, y_train)
    best_pipeline = search.best_estimator_

    predictions = best_pipeline.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    rep = classification_report(y_test, predictions, zero_division=0, output_dict=True)

    print(f"   [LR]  Acc: {acc:.4f} | F1: {rep['macro avg']['f1-score']:.4f} | "
          f"Prec: {rep['macro avg']['precision']:.4f} | Rec: {rep['macro avg']['recall']:.4f}")
    return best_pipeline, acc, rep


def train_evaluate_random_forest(X_train, y_train, X_test, y_test, seed=42):

    # default backend can deadlock on windows, n_jobs=1, -1 search to avoid
    param_dist_rf = {
        'n_estimators': [100, 150, 200, 300],
        'max_depth': [5, 7, 10, None],
        'min_samples_leaf':[2, 3, 5],
    }
    base_rf = RandomForestClassifier(
        random_state=seed, n_jobs=1, class_weight='balanced'
    )
    search_rf = RandomizedSearchCV(
        base_rf, param_dist_rf,
        n_iter=10, cv=3, scoring='f1_macro',
        random_state=seed, n_jobs=-1
    )
    search_rf.fit(X_train, y_train)
    rf_model = search_rf.best_estimator_

    train_preds = rf_model.predict(X_train)
    test_preds = rf_model.predict(X_test)

    train_acc = accuracy_score(y_train, train_preds)
    test_acc = accuracy_score(y_test, test_preds)
    rep = classification_report(y_test, test_preds, zero_division=0, output_dict=True)

    print(f"   [RF]  Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f} | "
          f"F1: {rep['macro avg']['f1-score']:.4f} | "
          f"Prec: {rep['macro avg']['precision']:.4f} | Rec: {rep['macro avg']['recall']:.4f}")


    return rf_model, test_acc, rep