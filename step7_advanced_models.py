"""
Step 7: Extend the model comparison with three stronger, more modern models:

1. GРADIENT BOOSTING (scikit-learn's GradientBoostingClassifier) -- builds
   trees one at a time, each correcting the errors of the previous ones.
   Usually stronger than a single Random Forest on small/medium tabular data.

2. XGBOOST -- a highly optimized, very popular gradient boosting library,
   often a top performer in real-world tabular ML competitions and papers.
   Needs a separate install: pip install xgboost --break-system-packages
   (or just: pip install xgboost)

3. NEURAL NETWORK (scikit-learn's MLPClassifier) -- a small feed-forward
   neural network. Included for completeness/comparison, though neural
   nets typically need MORE data than 395 rows to truly shine -- expect
   it to perform on par with or slightly worse than the tree-based models.

This script follows the EXACT same evaluation protocol as Step 6
(5-fold stratified cross-validation + GridSearchCV hyperparameter tuning)
so all 6 models (3 from before + 3 new ones) are directly comparable.

Make sure X_train.csv, X_test.csv, y_train.csv, y_test.csv (from Step 2)
are in the SAME FOLDER as this script, and that cv_tuned_results.csv
(from Step 6) is also there so we can merge results into one final table.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, recall_score

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("WARNING: xgboost not installed. Run: pip install xgboost")
    print("Continuing with Gradient Boosting and Neural Network only.\n")

# ------------------------------------------------------------------
# 1. Load data and recombine into one full dataset for CV (same as Step 6)
# ------------------------------------------------------------------
X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv").values.ravel()
y_test = pd.read_csv("y_test.csv").values.ravel()

X_full = pd.concat([X_train, X_test], ignore_index=True)
y_full = np.concatenate([y_train, y_test])

# Compute class weights manually for models that don't have class_weight='balanced'
# (XGBoost and MLP don't support that parameter the same way sklearn tree models do)
n_pass = (y_full == 1).sum()
n_fail = (y_full == 0).sum()
scale_pos_weight = n_fail / n_pass  # for XGBoost's imbalance handling

print(f"Full dataset: {X_full.shape[0]} students, {X_full.shape[1]} features")
print(f"Class balance -- pass: {n_pass}, fail: {n_fail}")
print()

# ------------------------------------------------------------------
# 2. Cross-validation setup (identical to Step 6)
# ------------------------------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fail_recall_scorer = make_scorer(recall_score, pos_label=0)
scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "fail_recall": fail_recall_scorer,
}

# ------------------------------------------------------------------
# 3. Define the new models and their hyperparameter grids
# ------------------------------------------------------------------
model_configs = {}

model_configs["Gradient Boosting"] = {
    "model": GradientBoostingClassifier(random_state=42),
    "params": {
        "n_estimators": [100, 200],
        "max_depth": [2, 3, 4],
        "learning_rate": [0.01, 0.1],
    },
}

if XGBOOST_AVAILABLE:
    model_configs["XGBoost"] = {
        "model": XGBClassifier(
            random_state=42, eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,  # handles class imbalance
        ),
        "params": {
            "n_estimators": [100, 200],
            "max_depth": [2, 3, 4],
            "learning_rate": [0.01, 0.1],
        },
    }

# Neural network needs scaled features to train properly -- wrap in a Pipeline
# so scaling happens correctly inside each cross-validation fold (avoids leakage)
model_configs["Neural Network (MLP)"] = {
    "model": Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(random_state=42, max_iter=2000)),
    ]),
    "params": {
        "mlp__hidden_layer_sizes": [(16,), (32,), (32, 16)],
        "mlp__alpha": [0.001, 0.01, 0.1],  # regularization strength
    },
}

# ------------------------------------------------------------------
# 4. Tune + cross-validate each new model (same logic as Step 6)
# ------------------------------------------------------------------
results = []
best_params_summary = []

for name, cfg in model_configs.items():
    print(f"=== {name} ===")

    grid_search = GridSearchCV(
        estimator=cfg["model"],
        param_grid=cfg["params"],
        scoring="f1",
        cv=cv,
        n_jobs=1,  # keep 1 for stability on Windows/IDLE, as in Step 6
    )
    grid_search.fit(X_full, y_full)

    best_model = grid_search.best_estimator_
    print("Best hyperparameters found:", grid_search.best_params_)
    best_params_summary.append({"Model": name, "Best Params": str(grid_search.best_params_)})

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
# 5. Combine with Step 6 results into one final 6-model comparison table
# ------------------------------------------------------------------
new_results_df = pd.DataFrame(results)

try:
    old_results_df = pd.read_csv("cv_tuned_results.csv")
    combined_df = pd.concat([old_results_df, new_results_df], ignore_index=True)
    print("=== Combined Final Summary: All 6 Models (5-fold CV, tuned) ===")
    print(combined_df.to_string(index=False))
    combined_df.to_csv("cv_tuned_results_all6.csv", index=False)
    print()
    print("Saved: cv_tuned_results_all6.csv (combined with Step 6 results)")
except FileNotFoundError:
    print("=== New Models Summary (Step 6 results not found to merge) ===")
    print(new_results_df.to_string(index=False))
    new_results_df.to_csv("cv_tuned_results_new_models.csv", index=False)
    print()
    print("Saved: cv_tuned_results_new_models.csv")
    print("(Could not find cv_tuned_results.csv from Step 6 to merge -- make sure it's in this folder)")

params_df = pd.DataFrame(best_params_summary)
params_df.to_csv("best_hyperparameters_new_models.csv", index=False)
print("Saved: best_hyperparameters_new_models.csv")
