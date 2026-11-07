"""
Step 4: Find which features matter most for predicting pass/fail.

We look at this two ways:
1. Random Forest feature importance -- how much each feature reduces
   "impurity" (confusion) across all the trees. Higher = more useful.
2. Logistic Regression coefficients -- shows DIRECTION too:
   positive coefficient = pushes toward "pass"
   negative coefficient = pushes toward "fail"

Make sure X_train.csv, X_test.csv, y_train.csv, y_test.csv (from Step 2)
are in the SAME FOLDER as this script.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# 1. Load the prepared data
X_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv").values.ravel()

feature_names = X_train.columns

# ============================================================
# PART 1: Random Forest importance
# ============================================================
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

rf_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": rf.feature_importances_
}).sort_values("Importance", ascending=False)

print("=== Top 10 features (Random Forest) ===")
print(rf_importance.head(10).to_string(index=False))
print()

# Plot top 10
plt.figure(figsize=(8, 5))
top10_rf = rf_importance.head(10)
plt.barh(top10_rf["Feature"][::-1], top10_rf["Importance"][::-1])
plt.xlabel("Importance")
plt.title("Top 10 Feature Importances (Random Forest)")
plt.tight_layout()
plt.savefig("rf_feature_importance.png", dpi=150)
print("Saved plot: rf_feature_importance.png")
print()

# ============================================================
# PART 2: Logistic Regression coefficients (shows direction)
# ============================================================
logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(X_train, y_train)

logreg_coef = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": logreg.coef_[0]
})

# Sort by absolute value to see strongest effects (positive or negative)
logreg_coef["AbsCoefficient"] = logreg_coef["Coefficient"].abs()
logreg_coef = logreg_coef.sort_values("AbsCoefficient", ascending=False)

print("=== Top 10 features (Logistic Regression coefficients) ===")
print("Positive = pushes toward PASS | Negative = pushes toward FAIL")
print(logreg_coef.head(10)[["Feature", "Coefficient"]].to_string(index=False))

# Save both tables for the paper
rf_importance.to_csv("rf_feature_importance.csv", index=False)
logreg_coef.to_csv("logreg_coefficients.csv", index=False)
print()
print("Saved: rf_feature_importance.csv, logreg_coefficients.csv")
