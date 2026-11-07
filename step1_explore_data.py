"""
Step 1: Load and explore the UCI Student Performance dataset.
Run this first to understand the data before any modeling.

Make sure student-mat.csv is in the SAME FOLDER as this script.
"""

import pandas as pd

# Load the Math dataset (note: this file uses semicolons, not commas)
df = pd.read_csv("student-mat.csv", sep=";")

# 1. Basic shape of the data
print("Number of students:", df.shape[0])
print("Number of columns:", df.shape[1])
print()

# 2. See the column names
print("Columns:")
print(df.columns.tolist())
print()

# 3. Peek at the first 5 rows
print("First 5 rows:")
print(df.head())
print()

# 4. Check for missing values
print("Missing values per column:")
print(df.isnull().sum())
print()

# 5. Basic statistics of the final grade (G3) -- this is what we'll predict
print("Final grade (G3) statistics:")
print(df["G3"].describe())
