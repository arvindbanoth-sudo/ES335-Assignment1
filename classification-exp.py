import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, KFold

np.random.seed(42)

# -----------------------------
# Generate dataset
# -----------------------------
X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2,
    random_state=1, n_clusters_per_class=2, class_sep=0.5
)

# Convert to DataFrame and Series
X = pd.DataFrame(X, columns=["feature1", "feature2"])
y = pd.Series(y)

# Plot dataset
plt.scatter(X["feature1"], X["feature2"], c=y)
plt.xlabel("feature1")
plt.ylabel("feature2")
plt.title("Generated dataset")
plt.show()

# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# -----------------------------
# Train custom decision tree
# -----------------------------
clf = DecisionTree(criterion="gini")  # or "information_gain" depending on your tree
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

# -----------------------------
# Evaluate metrics
# -----------------------------
print("Q2 (a) Results:")
print("Accuracy:", accuracy(y_test, y_pred))

# Per-class precision
classes = np.unique(y_test)
print("Precision per class:")
for c in classes:
    print(f"Class {c}:", precision(y_test, y_pred, c))

# Per-class recall
print("\nRecall per class:")
for c in classes:
    print(f"Class {c}:", recall(y_test, y_pred, c))

# -----------------------------
# Q2 (b) Cross-validation to find best depth
# -----------------------------
kf = KFold(n_splits=5, shuffle=True, random_state=42)

depths = range(1, 11)  # try depths 1 to 10
avg_scores = []

for d in depths:
    scores = []
    for train_idx, val_idx in kf.split(X):
        X_train_cv, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train_cv, y_val = y.iloc[train_idx], y.iloc[val_idx]

        clf_cv = DecisionTree(max_depth=d, criterion="information_gain")
        clf_cv.fit(X_train_cv, y_train_cv)
        y_pred_cv = clf_cv.predict(X_val)

        scores.append(accuracy(y_val, y_pred_cv))
    avg_scores.append(np.mean(scores))

best_depth = depths[np.argmax(avg_scores)]

print("\nQ2 (b) Results:")
print("Average accuracy for depths 1–10:", avg_scores)
print("Best depth:", best_depth)
print("Best CV accuracy:", max(avg_scores))
