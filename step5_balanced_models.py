"""
Step 5: Retrain models with class balancing.

Why: our data has 265 "pass" vs 130 "fail" (imbalanced). Models can get lazy
and just lean toward predicting "pass" since it's the majority class.

Fix: use class_weight="balanced", which tells the model to pay more
attention to the minority class (fail) during training. This doesn't
change the data itself -- it changes how much each mistake "costs"
during training.

Make sure X_train.csv, X_test.csv, y_train.csv, y_test.csv (from Step 2)
are in the SAME FOLDER as this script.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# 1. Load the prepared data
X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv").values.ravel()
y_test = pd.read_csv("y_test.csv").values.ravel()

# 2. Define models WITH class balancing this time
models = {
    "Logistic Regression (balanced)": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
    "Decision Tree (balanced)": DecisionTreeClassifier(random_state=42, class_weight="balanced"),
    "Random Forest (balanced)": RandomForestClassifier(random_state=42, class_weight="balanced"),
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Also calculate recall specifically for the "fail" class (label 0)
    # since that's the group we most want to catch
    fail_recall = recall_score(y_test, y_pred, pos_label=0)

    results.append({
        "Model": name,
        "Accuracy": round(acc, 3),
        "Precision (pass)": round(prec, 3),
        "Recall (pass)": round(rec, 3),
        "Recall (fail)": round(fail_recall, 3),
        "F1-score (pass)": round(f1, 3),
    })

    print(f"--- {name} ---")
    print("Confusion matrix (rows=actual, cols=predicted) [Fail, Pass]:")
    print(confusion_matrix(y_test, y_pred))
    print()

results_df = pd.DataFrame(results)
print("=== Summary comparison (balanced models) ===")
print(results_df.to_string(index=False))

results_df.to_csv("model_comparison_results_balanced.csv", index=False)
print()
print("Saved: model_comparison_results_balanced.csv")
print()
print("Compare this file to model_comparison_results.csv (from Step 3)")
print("to see the before/after effect of class balancing.")
