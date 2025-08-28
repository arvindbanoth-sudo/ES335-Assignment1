"""
Decision Tree implementation for both classification and regression
Supports:
    - discrete input, discrete output
    - real input, discrete output
    - discrete input, real output
    - real input, real output
"""

from dataclasses import dataclass
from typing import Literal, Union, Dict, Any

import numpy as np
import pandas as pd
from tree.utils import *


class Node:
    """
    Node class for representing decision tree nodes
    """

    def __init__(self, is_leaf=False, prediction=None,
                 feature=None, threshold=None, children=None):
        self.is_leaf = is_leaf
        self.prediction = prediction      # class label or mean value
        self.feature = feature            # feature name
        self.threshold = threshold        # threshold if continuous
        self.children = children or {}    # for categorical splits {value: Node}
        self.left = None                  # for threshold split
        self.right = None                 # for threshold split


@dataclass
class DecisionTree:
    criterion: Literal["information_gain", "gini_index", "mse"]  # mse for regression
    max_depth: int = 5

    def __init__(self, criterion="information_gain", max_depth=5):
        self.criterion = criterion
        self.max_depth = max_depth
        self.root = None

    def _build(self, X: pd.DataFrame, y: pd.Series, depth: int):
        """
        Recursive tree building function
        """

        # Stopping conditions
        if depth >= self.max_depth or len(np.unique(y)) == 1 or len(X) == 0:
            if check_ifreal(y):
                return Node(is_leaf=True, prediction=np.mean(y))
            else:
                return Node(is_leaf=True, prediction=y.mode()[0])

        # Find best attribute
        features = X.columns
        best_attr, best_val, best_gain = opt_split_attribute(X, y, self.criterion, features)

        if best_attr is None or best_gain <= 0:
            if check_ifreal(y):
                return Node(is_leaf=True, prediction=np.mean(y))
            else:
                return Node(is_leaf=True, prediction=y.mode()[0])

        node = Node(feature=best_attr, threshold=best_val)

        # Split data
        if best_val is None:  # discrete split
            splits = split_data(X, y, best_attr, None)
            for v, (Xv, yv) in splits.items():
                node.children[v] = self._build(Xv, yv, depth + 1)
        else:  # continuous split
            (X_left, y_left), (X_right, y_right) = split_data(X, y, best_attr, best_val)
            node.left = self._build(X_left, y_left, depth + 1)
            node.right = self._build(X_right, y_right, depth + 1)

        return node

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Build the decision tree using training data
        """
        self.root = self._build(X, y, depth=0)

    def _predict_one(self, x: pd.Series, node: Node):
        """
        Traverse the tree to predict for a single example
        """
        if node.is_leaf:
            return node.prediction

        if node.threshold is None:  # discrete
            val = x[node.feature]
            if val in node.children:
                return self._predict_one(x, node.children[val])
            else:
                # unseen category -> majority prediction at this node
                if node.children:
                    return list(node.children.values())[0].prediction
                return node.prediction
        else:  # continuous
            if x[node.feature] <= node.threshold:
                return self._predict_one(x, node.left)
            else:
                return self._predict_one(x, node.right)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict values for multiple test samples
        """
        preds = [self._predict_one(row, self.root) for _, row in X.iterrows()]
        return pd.Series(preds, index=X.index)

    def _print_tree(self, node: Node, indent=""):
        """
        Helper recursive function to print tree structure
        """
        if node.is_leaf:
            print(indent + "-> " + str(node.prediction))
            return

        if node.threshold is None:  # discrete
            for val, child in node.children.items():
                print(indent + f"?({node.feature} == {val})")
                self._print_tree(child, indent + "   ")
        else:  # continuous
            print(indent + f"?({node.feature} <= {node.threshold:.3f})")
            print(indent + "Y:")
            self._print_tree(node.left, indent + "   ")
            print(indent + "N:")
            self._print_tree(node.right, indent + "   ")

    def plot(self) -> None:
        """
        Print a text-based tree visualization
        """
        self._print_tree(self.root, indent="")
