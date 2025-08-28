import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

from tree.base import DecisionTree   # your custom implementation
from metrics import *   # if you have custom metric functions

np.random.seed(42)

# -----------------------
# Step 1: Load & clean data
# -----------------------
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
columns = ["mpg", "cylinders", "displacement", "horsepower", "weight",
           "acceleration", "model year", "origin", "car name"]

data = pd.read_csv(url, sep='\s+', header=None, names=columns)

# Replace '?' with NaN and drop missing rows
data = data.replace("?", np.nan)
data = data.dropna()

# Convert horsepower column to numeric (it may be object type)
data["horsepower"] = pd.to_numeric(data["horsepower"], errors="coerce")
data = data.dropna()


# Features and target
X = data.drop(["mpg", "car name"], axis=1)
y = data["mpg"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------
# Step 2: Custom Decision Tree
# -----------------------
my_tree = DecisionTree(max_depth=5)   # your implementation
my_tree.fit(X_train, y_train)
y_pred_my = my_tree.predict(X_test)

# Convert to numpy
y_pred_my = np.array(y_pred_my, dtype=float)

print("First 20 predictions from custom tree:", y_pred_my[:20])
print("Any NaN in predictions? ->", np.isnan(y_pred_my).any())
print("Length of predictions:", len(y_pred_my), " | Length of y_test:", len(y_test))

# -----------------------
# Step 3: Scikit-learn Decision Tree
# -----------------------
sk_tree = DecisionTreeRegressor(max_depth=5, random_state=42)
sk_tree.fit(X_train, y_train)
y_pred_sk = sk_tree.predict(X_test)

print("Scikit-learn Decision Tree:")
print("MSE:", mean_squared_error(y_test, y_pred_sk))
print("R²:", r2_score(y_test, y_pred_sk))
