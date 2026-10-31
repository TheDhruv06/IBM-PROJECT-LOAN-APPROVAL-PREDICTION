"""
=======================================================
 LOAN APPROVAL PREDICTION
 A simple machine learning project using scikit-learn
=======================================================

Goal   : Predict whether a loan application will be Approved (Y)
         or Rejected (N), based on applicant and financial details.
Type   : Supervised Learning -> Classification problem
         (because the output is a category, not a number)

Steps followed:
 1. Load the dataset
 2. Split into training data and test data
 3. Handle missing values
 4. Convert text columns into numbers
 5. Scale the numeric features
 6. Train 4 different models
 7. Compare their performance
 8. Find which features matter the most
=======================================================
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)


def section(title):
    """Prints a clean, readable section header in the console."""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


# -------------------------------------------------------------
# STEP 1: LOAD THE DATA
# -------------------------------------------------------------
section("STEP 1: LOADING THE DATASET")

df = pd.read_csv('loan.csv').drop(columns='Loan_ID')  # Loan_ID is just an ID, not useful for prediction

print(f"Total applications : {df.shape[0]}")
print(f"Total columns       : {df.shape[1]}")
print("Approved vs Rejected:")
print(df['Loan_Status'].value_counts().rename({'Y': 'Approved', 'N': 'Rejected'}).to_string())

print("\nMissing values per column:")
missing = df.isnull().sum()
missing = missing[missing > 0]
print(missing.to_string() if len(missing) else "  None")


# -------------------------------------------------------------
# STEP 2: SPLIT INTO TRAIN AND TEST DATA
# -------------------------------------------------------------
# We split BEFORE fixing missing values or scaling, so that the
# model never "sees" the test data in advance. This gives a fair,
# honest evaluation of accuracy (avoids data leakage).
# -------------------------------------------------------------
section("STEP 2: TRAIN / TEST SPLIT (70% / 30%)")

y = (df['Loan_Status'] == 'Y').astype(int)   # Y -> 1 (Approved), N -> 0 (Rejected)
X = df.drop(columns='Loan_Status')

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,       # 30% of data reserved for testing
    stratify=y,          # keeps the same Approved:Rejected ratio in both sets
    random_state=42       # makes the split reproducible (same result every run)
)

print(f"Training set size : {len(X_train)} rows")
print(f"Test set size     : {len(X_test)} rows")


# -------------------------------------------------------------
# STEP 3: HANDLE MISSING VALUES
# -------------------------------------------------------------
# Numeric columns  -> fill with the MEDIAN (robust to outliers)
# Categorical / Credit_History -> fill with the MODE (most frequent value)
#
# Important: the median/mode is calculated using ONLY the training
# data, then applied to both train and test. This avoids leaking
# information from the test set into training.
# -------------------------------------------------------------
section("STEP 3: HANDLING MISSING VALUES")

numeric_cols = ['LoanAmount', 'Loan_Amount_Term']

for col in X_train.columns:
    if col in numeric_cols:
        fill_value = X_train[col].median()
        method = "median"
    else:
        fill_value = X_train[col].mode()[0]
        method = "mode"

    if X_train[col].isnull().sum() > 0 or X_test[col].isnull().sum() > 0:
        print(f"  {col:<18} -> filled using {method} = {fill_value}")

    X_train[col] = X_train[col].fillna(fill_value)
    X_test[col] = X_test[col].fillna(fill_value)


# -------------------------------------------------------------
# STEP 4: CONVERT TEXT COLUMNS INTO NUMBERS (ONE-HOT ENCODING)
# -------------------------------------------------------------
# ML models only understand numbers, not text like "Male"/"Female".
# One-hot encoding turns each category into a 0/1 column.
# -------------------------------------------------------------
section("STEP 4: ENCODING CATEGORICAL COLUMNS")

X_train = pd.get_dummies(X_train, drop_first=True).astype(float)
X_test = pd.get_dummies(X_test, drop_first=True).astype(float)

# Make sure test set has exactly the same columns as train set
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

print(f"Final number of features after encoding: {X_train.shape[1]}")
print("Feature columns:", list(X_train.columns))


# -------------------------------------------------------------
# STEP 5: SCALE THE FEATURES
# -------------------------------------------------------------
# Logistic Regression and KNN are distance/gradient based, so they
# work better when all features are on a similar scale.
# Decision Tree and Random Forest do NOT need scaling.
# -------------------------------------------------------------
section("STEP 5: SCALING NUMERIC FEATURES")

scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Scaling done (mean = 0, standard deviation = 1).")


# -------------------------------------------------------------
# STEP 6: TRAIN AND EVALUATE MULTIPLE MODELS
# -------------------------------------------------------------
section("STEP 6: TRAINING MODELS")

# Each model: (algorithm, needs_scaled_data?)
models = {
    'Logistic Regression': (LogisticRegression(max_iter=1000), True),
    'KNN (k=5)':            (KNeighborsClassifier(n_neighbors=5), True),
    'Decision Tree':        (DecisionTreeClassifier(max_depth=3, random_state=42), False),
    'Random Forest':        (RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42), False),
}

results = {}
trained_models = {}

for name, (model, needs_scaling) in models.items():
    train_data = X_train_scaled if needs_scaling else X_train.values
    test_data = X_test_scaled if needs_scaling else X_test.values

    model.fit(train_data, y_train)
    predictions = model.predict(test_data)
    trained_models[name] = model

    results[name] = {
        'accuracy':  round(accuracy_score(y_test, predictions), 3),
        'precision': round(precision_score(y_test, predictions), 3),
        'recall':    round(recall_score(y_test, predictions), 3),
        'f1_score':  round(f1_score(y_test, predictions), 3),
        'confusion_matrix': confusion_matrix(y_test, predictions).tolist(),
    }
    print(f"  Trained: {name}")


# -------------------------------------------------------------
# STEP 7: DISPLAY RESULTS IN A CLEAN COMPARISON TABLE
# -------------------------------------------------------------
section("STEP 7: MODEL COMPARISON")

header = f"{'Model':<22}{'Accuracy':>10}{'Precision':>11}{'Recall':>9}{'F1-Score':>10}"
print(header)
print("-" * len(header))
for name, r in results.items():
    print(f"{name:<22}{r['accuracy']:>10}{r['precision']:>11}{r['recall']:>9}{r['f1_score']:>10}")

best_model_name = max(results, key=lambda n: results[n]['f1_score'])
print(f"\nBest model (highest F1-score): {best_model_name}")

cm = results[best_model_name]['confusion_matrix']
print(f"\nConfusion matrix for {best_model_name}:")
print(f"                    Predicted Rejected   Predicted Approved")
print(f"  Actual Rejected        {cm[0][0]:>4}                 {cm[0][1]:>4}")
print(f"  Actual Approved        {cm[1][0]:>4}                 {cm[1][1]:>4}")


# -------------------------------------------------------------
# STEP 8: FIND THE MOST IMPORTANT FACTORS
# -------------------------------------------------------------
section("STEP 8: MOST IMPORTANT FACTORS")

# Random Forest gives feature importance directly
rf_model = trained_models['Random Forest']
rf_importance = sorted(
    zip(X_train.columns, rf_model.feature_importances_),
    key=lambda pair: -pair[1]
)[:5]

print("Top 5 factors (Random Forest importance):")
for feature, score in rf_importance:
    print(f"  {feature:<25} {score:.3f}")

# Logistic Regression gives coefficients (direction + strength)
lr_model = trained_models['Logistic Regression']
lr_coef = sorted(
    zip(X_train.columns, lr_model.coef_[0]),
    key=lambda pair: -abs(pair[1])
)[:5]

print("\nTop 5 factors (Logistic Regression coefficients):")
for feature, coef in lr_coef:
    direction = "increases approval chance" if coef > 0 else "decreases approval chance"
    print(f"  {feature:<25} {coef:>6.2f}  ({direction})")

print("\nDone!")