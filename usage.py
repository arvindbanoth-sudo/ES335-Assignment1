"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

np.random.seed(42)


# --------------------------
# Test case 1: Real Input, Real Output
# --------------------------
print("\n===== Test Case 1: Real Input, Real Output =====")
N, P = 30, 5
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randn(N))

for criteria in ["mse"]:
    print(f"\n--- Criterion: {criteria} ---")
    tree = DecisionTree(criterion=criteria)
    tree.fit(X, y)
    y_hat = tree.predict(X)
    tree.plot()
    print("RMSE:", rmse(y_hat, y))
    print("MAE :", mae(y_hat, y))


# --------------------------
# Test case 2: Real Input, Discrete Output
# --------------------------
print("\n===== Test Case 2: Real Input, Discrete Output =====")
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randint(P, size=N), dtype="category")

for criteria in ["entropy", "gini"]:
    print(f"\n--- Criterion: {criteria} ---")
    tree = DecisionTree(criterion=criteria)
    tree.fit(X, y)
    y_hat = tree.predict(X)
    tree.plot()
    print("Accuracy:", accuracy(y_hat, y))
    for cls in y.unique():
        print(f"Precision (class {cls}):", precision(y_hat, y, cls))
        print(f"Recall (class {cls})   :", recall(y_hat, y, cls))


# --------------------------
# Test case 3: Discrete Input, Discrete Output
# --------------------------
print("\n===== Test Case 3: Discrete Input, Discrete Output =====")
X = pd.DataFrame({i: pd.Series(np.random.randint(P, size=N), dtype="category") for i in range(P)})
y = pd.Series(np.random.randint(P, size=N), dtype="category")

for criteria in ["entropy", "gini"]:
    print(f"\n--- Criterion: {criteria} ---")
    tree = DecisionTree(criterion=criteria)
    tree.fit(X, y)
    y_hat = tree.predict(X)
    tree.plot()
    print("Accuracy:", accuracy(y_hat, y))
    for cls in y.unique():
        print(f"Precision (class {cls}):", precision(y_hat, y, cls))
        print(f"Recall (class {cls})   :", recall(y_hat, y, cls))


# --------------------------
# Test case 4: Discrete Input, Real Output
# --------------------------
print("\n===== Test Case 4: Discrete Input, Real Output =====")
X = pd.DataFrame({i: pd.Series(np.random.randint(P, size=N), dtype="category") for i in range(P)})
y = pd.Series(np.random.randn(N))

for criteria in ["mse"]:
    print(f"\n--- Criterion: {criteria} ---")
    tree = DecisionTree(criterion=criteria)
    tree.fit(X, y)
    y_hat = tree.predict(X)
    tree.plot()
    print("RMSE:", rmse(y_hat, y))
    print("MAE :", mae(y_hat, y))
