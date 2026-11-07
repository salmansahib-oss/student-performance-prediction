# Predicting Student Academic Outcomes from Behavioral and Demographic Factors Excluding Prior Grades

Code accompanying the manuscript. The study predicts whether a secondary-school student passes (final grade G3 >= 10) or fails (G3 < 10) the Mathematics course, using only non-grade attributes. The first- and second-period grades (G1, G2) are deliberately excluded.

## Data

The scripts use the **Mathematics** file (`student-mat.csv`, 395 students, two Portuguese secondary schools) of the UCI *Student Performance* dataset (Cortez and Silva, 2008).

The data file is **not included** in this repository. Download it from the UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/320/student+performance

Place `student-mat.csv` in the same folder as the scripts. The file is semicolon-separated.

Note: the UCI page also hosts a Portuguese-language file (649 students). It is not used here.

## Requirements

Versions used in the study:

- Python 3.14.6
- scikit-learn 1.9.0
- XGBoost 3.3.0
- pandas, numpy, matplotlib

Install the packages with:

```
pip install pandas numpy matplotlib scikit-learn xgboost
```

## How to run

Run the scripts in order, from the folder that contains `student-mat.csv`. Each script prints its results and saves CSV files that later steps or the paper use.

| Script | What it does |
|---|---|
| `step1_explore_data.py` | Loads the data and prints its shape, columns, missing values, and G3 summary. |
| `step2_prepare_data.py` | Drops G1/G2, creates the pass/fail label, one-hot encodes categorical variables, and makes a stratified 80/20 split (`random_state=42`). Saves `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`. |
| `step3_train_models.py` | Single-split baseline: Logistic Regression, Decision Tree, Random Forest. |
| `step4_feature_importance.py` | Random Forest impurity importance and Logistic Regression coefficients (single fit). Saves `rf_feature_importance.png`. |
| `step5_balanced_models.py` | Repeats step 3 with `class_weight="balanced"`. |
| `step6_cv_tuning.py` | 5-fold stratified cross-validation with grid search for the three baseline models. |
| `step7_advanced_models.py` | Adds Gradient Boosting, XGBoost, and a Neural Network (MLP), with the same cross-validated tuning. |
| `step8_robust_evaluation.py` | Repeated 5x5 cross-validation with the tuned settings. Reports accuracy, balanced accuracy, sensitivity, specificity, precision, F1 (fail class), ROC-AUC, PR-AUC, out-of-fold confusion matrices, and permutation importance across folds. |

Step 8 needs the CSV files created by step 2.

## Notes

- Random seeds are fixed (`random_state=42`) for the split, all cross-validation schemes, and all stochastic models. Results may differ slightly with other library versions.
- On Windows, running grid search inside IDLE with `n_jobs=-1` can crash, so the scripts use `n_jobs=1`. This is slower but stable.
- The failing class is treated as the positive class in sensitivity, precision, F1, and PR-AUC.
- The study is an exploratory analysis of a small public dataset. It is not a validated early-warning system and should not be used for decisions about individual students.

## Reference

Cortez, P. and Silva, A.M.G., 2008. Using data mining to predict secondary school student performance. In: Proceedings of the 5th Annual Future Business Technology Conference (FUBUTEC), Porto, Portugal, pp.5-12.

## License

[Choose a license, for example MIT, when creating the repository.]
