import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report

# loguniform allows the search to explore C values exponentially (1e-3 to 10) 
# rather than linearly, which is appropriate for a regularisation parameter.
from scipy.stats import loguniform


def train_evaluate_logistic_regression(X_train, y_train, X_test, y_test):

    # switched from a fixed C=0.1 to using a Pipeline with StandardScaler and LogisticRegression
    # so the scaler is included when saving the model. I set up RandomizedSearchCV to tune C 
    # (log-uniform from 1e-3 to 10) and the solver, running 10 configs with 3-fold CV on f1_macro. 
    # using n_iter=10 keeps the search fast enough for the outer 5-fold loop.
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=2000, random_state=42, class_weight='balanced'))
    ])
    param_dist = {
        'clf__C':      loguniform(1e-3, 10),
        'clf__solver': ['lbfgs', 'saga'],
    }
    search = RandomizedSearchCV(
        pipeline, param_dist,
        n_iter=10, cv=3, scoring='f1_macro',
        random_state=42, n_jobs=-1
    )
    search.fit(X_train, y_train)
    best_pipeline = search.best_estimator_

    predictions = best_pipeline.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    rep = classification_report(y_test, predictions, zero_division=0, output_dict=True)

    print(f"   [LR]  Acc: {acc:.4f} | F1: {rep['macro avg']['f1-score']:.4f} | "
          f"Prec: {rep['macro avg']['precision']:.4f} | Rec: {rep['macro avg']['recall']:.4f}")
    return best_pipeline, acc, rep


def train_evaluate_random_forest(X_train, y_train, X_test, y_test, print_features=False):

    # implementation was using manually chosen dataset-agnostic hyperparameters so instead:
    # RandomizedSearchCV explores 10 random combinations of n_estimators, max_depth,and min_samples_leaf with 3-fold inner CV scored on f1_macro.
    # n_jobs=1 on the base RF and -1 on the search avoids nested-parallelism conflicts
    # that can deadlock on windows with the default loky backend.
    param_dist_rf = {
        'n_estimators': [100, 150, 200, 300],
        'max_depth': [5, 7, 10, None],
        'min_samples_leaf':[2, 3, 5],
    }
    base_rf = RandomForestClassifier(
        random_state=42, n_jobs=1, class_weight='balanced'
    )
    search_rf = RandomizedSearchCV(
        base_rf, param_dist_rf,
        n_iter=10, cv=3, scoring='f1_macro',
        random_state=42, n_jobs=-1
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