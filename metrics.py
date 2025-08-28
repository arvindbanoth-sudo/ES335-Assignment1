from typing import Union
import pandas as pd
import numpy as np

def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    """Classification Accuracy"""
    assert y_hat.size == y.size, "Predictions and labels must be same length"
    assert y.size > 0, "y cannot be empty"
    return (y_hat == y).sum() / y.size

def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """Precision for a given class"""
    assert y_hat.size == y.size
    tp = ((y_hat == cls) & (y == cls)).sum()
    fp = ((y_hat == cls) & (y != cls)).sum()
    if tp + fp == 0:
        return 0.0
    return tp / (tp + fp)

def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """Recall for a given class"""
    assert y_hat.size == y.size
    tp = ((y_hat == cls) & (y == cls)).sum()
    fn = ((y_hat != cls) & (y == cls)).sum()
    if tp + fn == 0:
        return 0.0
    return tp / (tp + fn)

def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    """Root Mean Squared Error"""
    assert y_hat.size == y.size
    return np.sqrt(((y_hat - y) ** 2).mean())

def mae(y_hat: pd.Series, y: pd.Series) -> float:
    """Mean Absolute Error"""
    assert y_hat.size == y.size
    return (y_hat - y).abs().mean()

import numpy as np

def misclassification_error(y):
    """
    Misclassification error = 1 - (proportion of most common class)
    """
    if len(y) == 0:
        return 0
    
    vals, counts = np.unique(y, return_counts=True)
    max_count = np.max(counts)
    return 1 - (max_count / len(y))
