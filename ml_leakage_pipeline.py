from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd


# Dataset
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)

# Task 1 — Data Leakage Example

# WRONG: scaling the data before splitting it. so the scaler can see the test data and memorize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc = accuracy_score(y_test, model.predict(X_test))

print(f"Train Accuracy (leaky): {train_acc:.4f}")
print(f"Test Accuracy (leaky): {test_acc:.4f}")

print("\nIssue:")
print("Scaling was applied BEFORE train-test split, meaning test data influenced training statistics (mean & std).")



# Task 2 — Pipeline + CV Fix

# Split first
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline ensures no leakage
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])

# Cross-validation
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5)

print(f"CV Mean Accuracy: {cv_scores.mean():.4f}")
print(f"CV Std Dev: {cv_scores.std():.4f}")

# Fit final model
pipeline.fit(X_train, y_train)

train_acc = accuracy_score(y_train, pipeline.predict(X_train))
test_acc = accuracy_score(y_test, pipeline.predict(X_test))

print(f"Train Accuracy (clean): {train_acc:.4f}")
print(f"Test Accuracy (clean): {test_acc:.4f}")


# =============================
# Task 3 — Decision Tree Depth
# =============================

depths = [1, 5, 20]
results = []

for depth in depths:
    tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
    tree.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, tree.predict(X_train))
    test_acc = accuracy_score(y_test, tree.predict(X_test))

    results.append({
        "max_depth": depth,
        "train_accuracy": train_acc,
        "test_accuracy": test_acc
    })

# Table creation
results_df = pd.DataFrame(results)
print(results_df)

print("\nAnalysis:")
print("- Depth 1: Underfitting (low train & test accuracy)")
print("- Depth 5: Good balance between bias and variance")
print("- Depth 20: Overfitting (very high train accuracy, lower test accuracy)")
