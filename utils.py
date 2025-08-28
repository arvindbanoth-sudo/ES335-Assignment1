"""
Utility functions for decision tree implementation
"""

import pandas as pd
import numpy as np

import numpy as np

def gini(y):
    """Compute Gini impurity for labels y"""
    _, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    return 1 - np.sum(probs ** 2)

def entropy(y):
    """Compute entropy for labels y"""
    _, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs + 1e-9))  # add small constant to avoid log(0)

def one_hot_encoding(X: pd.DataFrame) -> pd.DataFrame:
    """
    Perform one-hot encoding for discrete (categorical) features.
    If the input is already numeric, it returns X unchanged.
    """
    return pd.get_dummies(X)

def check_ifreal(y: pd.Series) -> bool:
    """
    Check if the target/output is real-valued (regression) or discrete (classification).
    Returns True if real-valued, False if discrete.
    """
    return pd.api.types.is_float_dtype(y) or pd.api.types.is_integer_dtype(y) and (len(y.unique()) > 10)

def entropy(Y: pd.Series) -> float:
    """
    Calculate entropy for classification targets
    """
    values, counts = np.unique(Y, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs + 1e-9))

def gini_index(Y: pd.Series) -> float:
    """
    Calculate Gini index for classification targets
    """
    values, counts = np.unique(Y, return_counts=True)
    probs = counts / counts.sum()
    return 1 - np.sum(probs ** 2)

def mse(Y: pd.Series) -> float:
    """
    Mean Squared Error for regression targets
    """
    mean = np.mean(Y)
    return np.mean((Y - mean) ** 2)

import numpy as np

import numpy as np

def misclassification_error(y):
    """
    Compute misclassification error impurity for labels y.
    Misclassification error = 1 - (proportion of most common class)
    """
    if len(y) == 0:
        return 0
    vals, counts = np.unique(y, return_counts=True)
    return 1 - (np.max(counts) / len(y))


def information_gain(y, col, criterion):
    if criterion in ["gini", "gini_index"]:
        parent_impurity = gini(y)
    elif criterion in ["entropy", "information_gain"]:
        parent_impurity = entropy(y)
    elif criterion in ["misclassification_error", "error"]:
        parent_impurity = misclassification_error(y)
    else:
        raise ValueError(f"Invalid criterion: {criterion}")

    vals, counts = np.unique(col, return_counts=True)
    weighted_impurity = 0.0
    for v, count in zip(vals, counts):
        subset = y[col == v]
        if criterion in ["gini", "gini_index"]:
            weighted_impurity += (count / len(y)) * gini(subset)
        elif criterion in ["entropy", "information_gain"]:
            weighted_impurity += (count / len(y)) * entropy(subset)
        elif criterion in ["misclassification_error", "error"]:
            weighted_impurity += (count / len(y)) * misclassification_error(subset)

    return parent_impurity - weighted_impurity


def opt_split_attribute(X: pd.DataFrame, y: pd.Series, criterion: str, features: pd.Series):
    """
    Find the attribute (and threshold if real) that gives the best information gain.
    Returns (best_attr, best_val, best_gain).
    """
    best_gain, best_attr, best_val = -1, None, None

    for attr in features:
        col = X[attr]
        values = np.unique(col)

        # If continuous, try threshold splits
        if pd.api.types.is_numeric_dtype(col) and len(values) > 10:
            thresholds = (values[:-1] + values[1:]) / 2
            for t in thresholds:
                left_y = y[col <= t]
                right_y = y[col > t]
                if len(left_y)==0 or len(right_y)==0:
                    continue
                if criterion == "mse":
                    gain = mse(y) - (len(left_y)/len(y))*mse(left_y) - (len(right_y)/len(y))*mse(right_y)
                else:
                    base = entropy(y) if criterion=="entropy" else gini_index(y)
                    gain = base - (len(left_y)/len(y))*(entropy(left_y) if criterion=="entropy" else gini_index(left_y)) \
                                 - (len(right_y)/len(y))*(entropy(right_y) if criterion=="entropy" else gini_index(right_y))
                if gain > best_gain:
                    best_gain, best_attr, best_val = gain, attr, t

        else:  # discrete feature
            gain = information_gain(y, col, criterion)
            if gain > best_gain:
                best_gain, best_attr, best_val = gain, attr, None

    return best_attr, best_val, best_gain

def split_data(X: pd.DataFrame, y: pd.Series, attribute, value):
    """
    Split the dataset on given attribute and value.
    Handles both discrete and continuous features.
    Returns left and right subsets of (X,y).
    """
    col = X[attribute]
    if value is None:  # discrete
        masks = {}
        for v in col.unique():
            masks[v] = (X[col == v].drop(columns=[attribute]), y[col == v])
        return masks
    else:  # continuous threshold
        left_mask = col <= value
        right_mask = col > value
        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[right_mask], y[right_mask]
        return (X_left, y_left), (X_right, y_right)
