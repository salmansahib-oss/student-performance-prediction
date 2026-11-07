"""
Step 2: Prepare the data for modeling.

What this does:
1. Loads the dataset
2. Drops G1 and G2 (so we predict G3 using ONLY social/demographic/behavioral factors)
3. Converts G3 into a pass/fail label (pass = G3 >= 10)
4. Converts categorical (text) columns into numbers, since ML models need numbers
5. Splits data into training set (80%) and testing set (20%)

Make sure student-mat.csv is in the SAME FOLDER as this script.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Load data
df = pd.read_csv("student-mat.csv", sep=";")

# 2. Drop G1 and G2 (avoid leaking prior grades into the prediction)
df = df.drop(columns=["G1", "G2"])

# 3. Create pass/fail label from G3
#    pass = 1 if G3 >= 10, fail = 0 if G3 < 10
df["pass"] = (df["G3"] >= 10).astype(int)

# Now drop G3 itself, since "pass" is our new target
df = df.drop(columns=["G3"])

# Quick check: how balanced is pass vs fail?
print("Pass/Fail counts:")
print(df["pass"].value_counts())
print()

# 4. Convert categorical (text) columns into numeric columns
#    Example: sex ('F'/'M') becomes sex_M (0 or 1)
categorical_cols = df.select_dtypes(include="object").columns.tolist()
print("Categorical columns being encoded:", categorical_cols)
print()

df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

print("Shape after encoding:", df_encoded.shape)
print()

# 5. Split into features (X) and target (y)
X = df_encoded.drop(columns=["pass"])
y = df_encoded["pass"]

# 6. Train/test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training set size:", X_train.shape[0])
print("Testing set size:", X_test.shape[0])

# Save the prepared data so Step 3 can use it directly
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print()
print("Saved: X_train.csv, X_test.csv, y_train.csv, y_test.csv")
