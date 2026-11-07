"""
Step 3: Train and compare 3 machine learning models.

Models used:
1. Logistic Regression -- simple, interpretable baseline
2. Decision Tree -- easy to visualize, shows clear if/else rules
3. Random Forest -- combines many decision trees, usually more accurate

For each model we report:
- Accuracy: overall % correct
- Precision: of students predicted to PASS, how many actually passed
- Recall: of students who actually passed, how many did we correctly catch
- F1-score: balance of precision and recall

Make sure X_train.csv, X_test.csv, y_train.csv, y_test.csv (from Step 2)
are in the SAME FOLDER as this script.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# 1. Load the prepared data from Step 2
X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv").values.ravel()  # ravel: turn column into simple list
y_test = pd.read_csv("y_test.csv").values.ravel()

# 2. Define the models we want to compare
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
}

# 3. Train each model and evaluate it
results = []

for name, model in models.items():
    # Train (fit) the model on training data
    model.fit(X_train, y_train)

    # Predict on the unseen test data
    y_pred = model.predict(X_test)

    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results.append({
        "Model": name,
        "Accuracy": round(acc, 3),
        "Precision": round(prec, 3),
        "Recall": round(rec, 3),
        "F1-score": round(f1, 3),
    })

    print(f"--- {name} ---")
    print("Confusion matrix (rows=actual, cols=predicted) [Fail, Pass]:")
    print(confusion_matrix(y_test, y_pred))
    print()

# 4. Show a summary comparison table
results_df = pd.DataFrame(results)
print("=== Summary comparison ===")
print(results_df.to_string(index=False))

# Save results for the paper's results table
results_df.to_csv("model_comparison_results.csv", index=False)
print()
print("Saved: model_comparison_results.csv")
