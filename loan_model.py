"""
=======================================================
 LOAN APPROVAL PREDICTION - FINAL MODEL
 Logistic Regression only
=======================================================

This is our FINAL model, based on what we learned from
comparing 4 algorithms earlier (loan_model.py). Logistic
Regression gave the best F1-score and is also the easiest
to explain, so we are going ahead with it.

Steps:
 1. Load the dataset
 2. Split into training data and test data
 3. Handle missing values
 4. Convert text columns into numbers
 5. Scale the features
 6. Train the Logistic Regression model
 7. Evaluate it
 8. Explain which features matter the most
=======================================================
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
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


# -------------------------------------------------------------
# STEP 2: SPLIT INTO TRAIN AND TEST DATA
# -------------------------------------------------------------
# Split BEFORE cleaning, so the test set stays completely unseen.
# (Same reasoning as our comparison script: avoids data leakage.)
# -------------------------------------------------------------
section("STEP 2: TRAIN / TEST SPLIT (70% / 30%)")

y = (df['Loan_Status'] == 'Y').astype(int)   # Y -> 1 (Approved), N -> 0 (Rejected)
X = df.drop(columns='Loan_Status')

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    stratify=y,
    random_state=42
)

print(f"Training set size : {len(X_train)} rows")
print(f"Test set size     : {len(X_test)} rows")


# -------------------------------------------------------------
# STEP 3: HANDLE MISSING VALUES
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
section("STEP 4: ENCODING CATEGORICAL COLUMNS")

X_train = pd.get_dummies(X_train, drop_first=True).astype(float)
X_test = pd.get_dummies(X_test, drop_first=True).astype(float)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

print(f"Final number of features after encoding: {X_train.shape[1]}")
print("Feature columns:", list(X_train.columns))


# -------------------------------------------------------------
# STEP 5: SCALE THE FEATURES
# -------------------------------------------------------------
# Logistic Regression uses gradient-based optimization, so it
# converges faster and more reliably when features are scaled
# to a similar range (mean 0, standard deviation 1).
# -------------------------------------------------------------
section("STEP 5: SCALING FEATURES")

scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Scaling done (mean = 0, standard deviation = 1).")


# -------------------------------------------------------------
# STEP 6: TRAIN THE LOGISTIC REGRESSION MODEL
# -------------------------------------------------------------
section("STEP 6: TRAINING LOGISTIC REGRESSION")

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)
predictions = model.predict(X_test_scaled)

print("Model trained successfully.")


# -------------------------------------------------------------
# STEP 7: EVALUATE THE MODEL
# -------------------------------------------------------------
section("STEP 7: MODEL EVALUATION")

acc = accuracy_score(y_test, predictions)
prec = precision_score(y_test, predictions)
rec = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)
cm = confusion_matrix(y_test, predictions)

print(f"Accuracy  : {acc:.3f}")
print(f"Precision : {prec:.3f}")
print(f"Recall    : {rec:.3f}")
print(f"F1-Score  : {f1:.3f}")

print("\nConfusion matrix:")
print(f"                    Predicted Rejected   Predicted Approved")
print(f"  Actual Rejected        {cm[0][0]:>4}                 {cm[0][1]:>4}")
print(f"  Actual Approved        {cm[1][0]:>4}                 {cm[1][1]:>4}")

print("\nFull classification report:")
print(classification_report(y_test, predictions, target_names=['Rejected', 'Approved']))


# -------------------------------------------------------------
# STEP 8: EXPLAIN THE MODEL (WHICH FEATURES MATTER MOST)
# -------------------------------------------------------------
section("STEP 8: MOST IMPORTANT FACTORS")

coefficients = sorted(
    zip(X_train.columns, model.coef_[0]),
    key=lambda pair: -abs(pair[1])
)

print("All feature coefficients (sorted by strength):")
for feature, coef in coefficients:
    direction = "increases approval chance" if coef > 0 else "decreases approval chance"
    print(f"  {feature:<25} {coef:>6.2f}  ({direction})")

print("\nDone!")