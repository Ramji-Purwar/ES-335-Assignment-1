"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tree.utils import *

np.random.seed(42)


@dataclass
class DecisionTree:
    criterion: Literal["information_gain", "gini_index"]  # criterion won't be used for regression
    max_depth: int  # The maximum depth the tree can grow to

    def __init__(self, criterion, max_depth=5):
        self.criterion = criterion
        self.max_depth = max_depth
        self.tree = None
        self.is_regression = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Function to train and construct the decision tree
        """
        # Determine if this is regression or classification
        self.is_regression = check_ifreal(y) and len(y.unique()) > 10
        
        # Build the tree
        self.tree = self._build_tree(X, y, 0)
    
    def _build_tree(self, X, y, depth):
        """Recursively build the decision tree"""
        # Stopping criteria
        if (depth >= self.max_depth or 
            len(y) <= 1 or 
            len(y.unique()) == 1 or 
            len(X.columns) == 0):
            
            # Create leaf node
            if self.is_regression:
                prediction = y.mean()
            else:
                prediction = y.mode().iloc[0] if len(y.mode()) > 0 else y.iloc[0]
            
            return {
                'type': 'leaf',
                'prediction': prediction
            }
        
        # Determine criterion
        if self.is_regression:
            criterion_name = "mse"
        else:
            criterion_name = "entropy" if self.criterion == "information_gain" else "gini_index"
        
        # Find best split
        best_feature = None
        best_threshold = None
        best_gain = -np.inf
        
        for feature in X.columns:
            if pd.api.types.is_numeric_dtype(X[feature]) and not pd.api.types.is_categorical_dtype(X[feature]):
                # Handle continuous features
                values = sorted(X[feature].unique())
                if len(values) > 1:
                    for i in range(len(values) - 1):
                        threshold = (values[i] + values[i + 1]) / 2
                        
                        # Create binary split
                        left_mask = X[feature] <= threshold
                        right_mask = ~left_mask
                        
                        if left_mask.sum() == 0 or right_mask.sum() == 0:
                            continue
                        
                        # Calculate information gain
                        binary_attr = pd.Series(['left' if x else 'right' for x in left_mask], index=y.index)
                        gain = information_gain(y, binary_attr, criterion_name)
                        
                        if gain > best_gain:
                            best_gain = gain
                            best_feature = feature
                            best_threshold = threshold
            else:
                # Handle discrete features
                if len(X[feature].unique()) > 1:
                    gain = information_gain(y, X[feature], criterion_name)
                    if gain > best_gain:
                        best_gain = gain
                        best_feature = feature
                        best_threshold = None
        
        # If no good split found, create leaf
        if best_feature is None or best_gain <= 0:
            if self.is_regression:
                prediction = y.mean()
            else:
                prediction = y.mode().iloc[0] if len(y.mode()) > 0 else y.iloc[0]
            
            return {
                'type': 'leaf',
                'prediction': prediction
            }
        
        # Create internal node
        node = {
            'type': 'internal',
            'feature': best_feature,
            'threshold': best_threshold,
            'children': {}
        }
        
        # Create children
        if best_threshold is not None:
            # Binary split for continuous
            left_mask = X[best_feature] <= best_threshold
            right_mask = ~left_mask
            
            if left_mask.sum() > 0:
                node['children']['left'] = self._build_tree(X[left_mask], y[left_mask], depth + 1)
            if right_mask.sum() > 0:
                node['children']['right'] = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        else:
            # Multi-way split for discrete
            for value in X[best_feature].unique():
                mask = X[best_feature] == value
                if mask.sum() > 0:
                    node['children'][value] = self._build_tree(X[mask], y[mask], depth + 1)
        
        return node

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Funtion to run the decision tree on test inputs
        """
        if self.tree is None:
            raise ValueError("Tree has not been fitted yet")
        
        predictions = []
        for _, row in X.iterrows():
            pred = self._predict_single(row, self.tree)
            predictions.append(pred)
        
        return pd.Series(predictions, index=X.index)
    
    def _predict_single(self, row, node):
        """Predict single instance"""
        if node['type'] == 'leaf':
            return node['prediction']
        
        feature = node['feature']
        threshold = node['threshold']
        
        if threshold is not None:
            # Continuous feature
            if row[feature] <= threshold:
                if 'left' in node['children']:
                    return self._predict_single(row, node['children']['left'])
                else:
                    # Fallback if no left child
                    return node.get('prediction', 0 if self.is_regression else 'Unknown')
            else:
                if 'right' in node['children']:
                    return self._predict_single(row, node['children']['right'])
                else:
                    # Fallback if no right child
                    return node.get('prediction', 0 if self.is_regression else 'Unknown')
        else:
            # Discrete feature
            value = row[feature]
            if value in node['children']:
                return self._predict_single(row, node['children'][value])
            else:
                # Fallback for unseen values
                return node.get('prediction', 0 if self.is_regression else 'Unknown')

    def plot(self) -> None:
        """
        Function to plot the tree

        Output Example:
        ?(X1 > 4)
            Y: ?(X2 > 7)
                Y: Class A
                N: Class B
            N: Class C
        Where Y => Yes and N => No
        """
        if self.tree is None:
            print("Tree has not been fitted yet")
            return
        
        print("Decision Tree:")
        self._plot_tree(self.tree, "")
    
    def _plot_tree(self, node, prefix):
        """Recursively plot the tree"""
        if node['type'] == 'leaf':
            print(f"{prefix}Prediction: {node['prediction']}")
            return
        
        feature = node['feature']
        threshold = node['threshold']
        
        if threshold is not None:
            # Continuous feature
            print(f"{prefix}?({feature} <= {threshold:.2f})")
            if 'left' in node['children']:
                print(f"{prefix}  Y:", end=" ")
                self._plot_tree(node['children']['left'], prefix + "    ")
            if 'right' in node['children']:
                print(f"{prefix}  N:", end=" ")
                self._plot_tree(node['children']['right'], prefix + "    ")
        else:
            # Discrete feature
            print(f"{prefix}?({feature})")
            for value, child in node['children'].items():
                print(f"{prefix}  {value}:", end=" ")
                self._plot_tree(child, prefix + "    ")
