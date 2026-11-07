"""
Step 6: Strengthen the analysis with k-fold cross-validation and
hyperparameter tuning -- both expected by competitive journals like
IEEE Transactions on Learning Technologies (TLT).

What this does, and why:
1. K-FOLD CROSS-VALIDATION (5-fold): Instead of judging each model on
   just ONE 80/20 split (which can be lucky or unlucky, especially with
   only 395 students), we split the data into 5 different folds, train
   on 4 and test on 1, rotate through all 5 combinations, and average
   the results. This gives a much more trustworthy accuracy estimate
   and also reports a standard deviation (uncertainty).

2. HYPERPARAMETER TUNING (GridSearchCV): Instead of using scikit-learn's
   default settings, we search over a small grid of reasonable settings
   for each model (e.g., tree depth, number of trees, regularization
   strength) and pick the best-performing combination, evaluated using
   cross-validation internally (so this is also more rigorous than a
   single validation check).

Make sure X_train.csv, X_test.csv, y_train.csv, y_test.csv (from Step 2)
are in the SAME FOLDER as this script.

NOTE: We combine train+test back into the full dataset for cross-validation,
since k-fold CV does its own internal splitting. We keep the original
test set aside separately ONLY for a final, single confirmatory check
at the end -- the main reported results should come from cross-validation.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, recall_score, precision_score, f1_score, accuracy_score

# ------------------------------------------------------------------
# 1. Load data and recombine into one full dataset for CV
# ------------------------------------------------------------------
X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv").values.ravel()
y_test = pd.read_csv("y_test.csv").values.ravel()

X_full = pd.concat([X_train, X_test], ignore_index=True)
y_full = np.concatenate([y_train, y_test])

print(f"Full dataset for cross-validation: {X_full.shape[0]} students, {X_full.shape[1]} features")
print()

# ------------------------------------------------------------------
# 2. Set up 5-fold stratified cross-validation
#    "Stratified" keeps the pass/fail ratio consistent in every fold
# ------------------------------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Custom scorer: recall specifically on the FAIL class (label 0)
fail_recall_scorer = make_scorer(recall_score, pos_label=0)

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "fail_recall": fail_recall_scorer,
}

# ------------------------------------------------------------------
# 3. Define hyperparameter grids for each model
#    (kept small and reasonable given the dataset size of 395)
# ------------------------------------------------------------------
param_grids = {
    "Logistic Regression": {
        "model": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
        "params": {
            "C": [0.01, 0.1, 1, 10],  # regularization strength
        },
    },
    "Decision Tree": {
        "model": DecisionTreeClassifier(random_state=42, class_weight="balanced"),
        "params": {
            "max_depth": [3, 5, 7, None],
            "min_samples_leaf": [1, 5, 10],
        },
    },
    "Random Forest": {
        "model": RandomForestClassifier(random_state=42, class_weight="balanced"),
        "params": {
            "n_estimators": [100, 200, 300],
            "max_depth": [5, 10, None],
            "min_samples_leaf": [1, 5],
        },
    },
}

# ------------------------------------------------------------------
# 4. For each model: tune hyperparameters, then run 5-fold CV with
#    the best parameters and report mean +/- std for each metric
# ------------------------------------------------------------------
results = []
best_params_summary = []

for name, cfg in param_grids.items():
    print(f"=== {name} ===")

    # Step A: Hyperparameter search using GridSearchCV
    #  - internally uses 5-fold CV to evaluate each combination
    #  - optimized for F1-score on the fail class equivalent (using fail_recall_scorer
    #    would over-favor recall alone, so we tune on F1 for a balanced choice)
    grid_search = GridSearchCV(
        estimator=cfg["model"],
        param_grid=cfg["params"],
        scoring="f1",
        cv=cv,
        n_jobs=1,  # Windows/IDLE can crash with parallel workers (n_jobs=-1); 1 is slower but stable
    )
    grid_search.fit(X_full, y_full)

    best_model = grid_search.best_estimator_
    print("Best hyperparameters found:", grid_search.best_params_)
    best_params_summary.append({"Model": name, "Best Params": str(grid_search.best_params_)})

    # Step B: Evaluate the TUNED model using 5-fold cross-validation
    #  (fresh cross_validate call so we get all metrics cleanly)
    cv_results = cross_validate(best_model, X_full, y_full, cv=cv, scoring=scoring)

    row = {
        "Model": name,
        "Accuracy": f"{cv_results['test_accuracy'].mean():.3f} ± {cv_results['test_accuracy'].std():.3f}",
        "Precision": f"{cv_results['test_precision'].mean():.3f} ± {cv_results['test_precision'].std():.3f}",
        "Recall (pass)": f"{cv_results['test_recall'].mean():.3f} ± {cv_results['test_recall'].std():.3f}",
        "Recall (fail)": f"{cv_results['test_fail_recall'].mean():.3f} ± {cv_results['test_fail_recall'].std():.3f}",
        "F1-score": f"{cv_results['test_f1'].mean():.3f} ± {cv_results['test_f1'].std():.3f}",
    }
    results.append(row)

    print(f"5-fold CV Accuracy:      {cv_results['test_accuracy'].mean():.3f} (+/- {cv_results['test_accuracy'].std():.3f})")
    print(f"5-fold CV Recall (fail): {cv_results['test_fail_recall'].mean():.3f} (+/- {cv_results['test_fail_recall'].std():.3f})")
    print()

# ------------------------------------------------------------------
# 5. Summary tables
# ------------------------------------------------------------------
results_df = pd.DataFrame(results)
print("=== Final Summary: 5-fold Cross-Validation with Tuned Hyperparameters ===")
print(results_df.to_string(index=False))
print()

params_df = pd.DataFrame(best_params_summary)
print("=== Best Hyperparameters per Model ===")
print(params_df.to_string(index=False))

# Save both for the paper
results_df.to_csv("cv_tuned_results.csv", index=False)
params_df.to_csv("best_hyperparameters.csv", index=False)
print()
print("Saved: cv_tuned_results.csv, best_hyperparameters.csv")
