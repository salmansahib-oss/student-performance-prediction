"""
Step 8: Address a doctoral reviewer's methodological concerns.

The reviewer asked for, at minimum:
  - confusion matrices
  - specificity, balanced accuracy, ROC-AUC, PR-AUC (not just accuracy/recall)
  - confidence intervals / repeated cross-validation variability (not just
    a single 5-fold run)
  - permutation importance, to check whether "failures", "goout", etc. are
    STABLE predictors across folds, not an artifact of one model fit

This script reuses the best hyperparameters already found by grid search
in Step 6/Step 7 (hardcoded below, so we don't have to repeat the slow
search), and uses them for a more rigorous, repeated evaluation.

Make sure X_train.csv, X_test.csv, y_train.csv, y_test.csv (from Step 2)
are in the SAME FOLDER as this script.

Needs: pip install xgboost  (if not already installed from Step 7)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, cross_validate, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    make_scorer, recall_score, precision_score, f1_score, balanced_accuracy_score,
    confusion_matrix, average_precision_score, roc_auc_score,
)
from sklearn.inspection import permutation_importance

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("WARNING: xgboost not installed. Run: pip install xgboost")

# ------------------------------------------------------------------
# 1. Load data (recombine train+test, as in Steps 6-7 -- CV does its own
#    splitting, so using the full 395 rows gives every fold more data)
# ------------------------------------------------------------------
X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv").values.ravel()
y_test = pd.read_csv("y_test.csv").values.ravel()

X_full = pd.concat([X_train, X_test], ignore_index=True)
y_full = np.concatenate([y_train, y_test])
# Recall: pass = 1, fail = 0 (set in Step 2). "Fail" is the class we most
# care about catching, so we treat it as the clinically/educationally
# "positive" class throughout this script (sensitivity = recall on fail).

n_pass = (y_full == 1).sum()
n_fail = (y_full == 0).sum()
scale_pos_weight = n_fail / n_pass

print(f"Full dataset: {X_full.shape[0]} students, {X_full.shape[1]} features")
print(f"pass={n_pass}, fail={n_fail}")
print()

# ------------------------------------------------------------------
# 2. Best hyperparameters already found in Step 6 / Step 7 (grid search).
#    We reuse them here instead of re-tuning, so that repeated CV runs
#    in a reasonable amount of time on a laptop.
# ------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(
        C=0.01, max_iter=1000, random_state=42, class_weight="balanced"),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=3, min_samples_leaf=5, random_state=42, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, max_depth=None, min_samples_leaf=1,
        random_state=42, class_weight="balanced"),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200, max_depth=2, learning_rate=0.01, random_state=42),
    "Neural Network (MLP)": Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(hidden_layer_sizes=(32,), alpha=0.001,
                               random_state=42, max_iter=2000)),
    ]),
}
if XGBOOST_AVAILABLE:
    models["XGBoost"] = XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.01,
        random_state=42, eval_metric="logloss", scale_pos_weight=scale_pos_weight)

# ------------------------------------------------------------------
# 3. Custom scorers for the metrics the reviewer specifically asked for
# ------------------------------------------------------------------
sensitivity_scorer = make_scorer(recall_score, pos_label=0)       # recall on FAIL
specificity_scorer = make_scorer(recall_score, pos_label=1)       # recall on PASS
precision_fail_scorer = make_scorer(precision_score, pos_label=0)
f1_fail_scorer = make_scorer(f1_score, pos_label=0)

def pr_auc_fail_scorer(estimator, X, y_true):
    # Average precision computed against the FAIL class specifically
    # (not the default "pass" positive class, which would look
    # artificially good simply because pass is the majority class).
    # Written as a plain function (estimator, X, y) so it works on every
    # scikit-learn version (make_scorer's needs_proba was removed in 1.6).
    proba = estimator.predict_proba(X)
    fail_col = list(estimator.classes_).index(0)
    return average_precision_score((y_true == 0).astype(int), proba[:, fail_col])

scoring = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "sensitivity_fail": sensitivity_scorer,
    "specificity_pass": specificity_scorer,
    "precision_fail": precision_fail_scorer,
    "f1_fail": f1_fail_scorer,
    "roc_auc": "roc_auc",
    "pr_auc_fail": pr_auc_fail_scorer,
}

# ------------------------------------------------------------------
# 4. ROBUST EVALUATION: repeated 5-fold CV (5 repeats = 25 estimates
#    per model instead of just 5), giving a real basis for confidence
#    intervals, as the reviewer requested.
# ------------------------------------------------------------------
rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)

results = []
for name, model in models.items():
    print(f"=== {name}: repeated 5x5 cross-validation ===")
    cv_res = cross_validate(model, X_full, y_full, cv=rcv, scoring=scoring, n_jobs=1)

    row = {"Model": name}
    for metric in scoring.keys():
        vals = cv_res[f"test_{metric}"]
        mean = vals.mean()
        std = vals.std()
        # approximate 95% CI using the 25 repeated-CV estimates
        ci95 = 1.96 * std / np.sqrt(len(vals))
        row[metric] = f"{mean:.3f} ± {std:.3f} (95% CI ≈ ±{ci95:.3f})"
        print(f"  {metric:20s}: {mean:.3f} ± {std:.3f}  (n={len(vals)} folds)")
    results.append(row)
    print()

results_df = pd.DataFrame(results)
results_df.to_csv("robust_evaluation_results.csv", index=False)
print("Saved: robust_evaluation_results.csv")
print()

# ------------------------------------------------------------------
# 5. CONFUSION MATRICES: one set of out-of-fold predictions per model,
#    using a single 5-fold split (every student predicted exactly once
#    by a model that never saw them in training).
# ------------------------------------------------------------------
cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("=== Confusion matrices (out-of-fold predictions, 5-fold) ===")
cm_rows = []
for name, model in models.items():
    y_pred = cross_val_predict(model, X_full, y_full, cv=cv5, n_jobs=1)
    cm = confusion_matrix(y_full, y_pred, labels=[0, 1])  # rows/cols = [fail, pass]
    tn_fail, fp_fail, fn_fail, tp_pass = cm.ravel()
    # relabel clearly: cm[0,0]=fail correctly caught, cm[0,1]=fail missed (predicted pass)
    #                   cm[1,0]=pass wrongly flagged fail, cm[1,1]=pass correctly caught
    print(f"{name}:")
    print(f"  [[fail->fail={cm[0,0]}, fail->pass={cm[0,1]}], [pass->fail={cm[1,0]}, pass->pass={cm[1,1]}]]")
    cm_rows.append({
        "Model": name,
        "Fail correctly caught": cm[0, 0],
        "Fail missed (predicted pass)": cm[0, 1],
        "Pass wrongly flagged fail": cm[1, 0],
        "Pass correctly caught": cm[1, 1],
    })
print()
cm_df = pd.DataFrame(cm_rows)
cm_df.to_csv("confusion_matrices.csv", index=False)
print("Saved: confusion_matrices.csv")
print()

# ------------------------------------------------------------------
# 6. PERMUTATION IMPORTANCE across folds, for the two models already
#    discussed in the paper's feature-importance tables (Random Forest,
#    Logistic Regression). This checks whether "failures", "goout", etc.
#    are STABLE predictors, not an artifact of a single fit -- directly
#    answering the reviewer's concern about feature-importance stability.
# ------------------------------------------------------------------
print("=== Permutation importance stability across 5 folds ===")
feature_names = X_full.columns

for name in ["Logistic Regression", "Random Forest"]:
    model = models[name]
    print(f"--- {name} ---")
    fold_importances = []

    for fold_i, (train_idx, val_idx) in enumerate(cv5.split(X_full, y_full)):
        X_tr, X_val = X_full.iloc[train_idx], X_full.iloc[val_idx]
        y_tr, y_val = y_full[train_idx], y_full[val_idx]

        model.fit(X_tr, y_tr)
        result = permutation_importance(
            model, X_val, y_val, n_repeats=10, random_state=42, scoring="f1"
        )
        fold_importances.append(result.importances_mean)

    fold_importances = np.array(fold_importances)  # shape (5 folds, n_features)
    mean_importance = fold_importances.mean(axis=0)
    std_importance = fold_importances.std(axis=0)

    imp_df = pd.DataFrame({
        "Feature": feature_names,
        "Mean Importance": mean_importance,
        "Std Importance": std_importance,
    }).sort_values("Mean Importance", ascending=False)

    print(imp_df.head(10).to_string(index=False))
    print()

    safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
    imp_df.to_csv(f"permutation_importance_{safe_name}.csv", index=False)
    print(f"Saved: permutation_importance_{safe_name}.csv")
    print()

print("=== Done. Files saved: robust_evaluation_results.csv, confusion_matrices.csv,")
print("    permutation_importance_Logistic_Regression.csv, permutation_importance_Random_Forest.csv ===")
