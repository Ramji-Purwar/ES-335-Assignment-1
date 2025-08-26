"""
You can add your own functions here according to your decision tree implementation.
There is no restriction on following the below template, these fucntions are here to simply help you.
"""

import numpy as np
import pandas as pd

def one_hot_encoding(X: pd.DataFrame) -> pd.DataFrame:
    """
    Function to perform one hot encoding on the input data
    """
    X_encoded = pd.get_dummies(X, drop_first=True)
    return X_encoded


def check_ifreal(y: pd.Series) -> bool:
    """
    Function to check if the given series has real or discrete values
    """
    if pd.api.types.is_numeric_dtype(y):
        return True
    return False


def entropy(Y: pd.Series) -> float:
    """
    Function to calculate the entropy
    """
    value_cnt = Y.value_counts()
    probabilities = value_cnt / len(Y)
    prob = probabilities.to_numpy()

    entropy_val = -np.sum(prob * np.log2(prob + 1e-9))

    return entropy_val


def gini_index(Y: pd.Series) -> float:
    """
    Function to calculate the gini index
    """
    value_cnt = Y.value_counts()
    probabilities = value_cnt / len(Y)
    prob = probabilities.to_numpy()

    gini_indexx = 1 - np.sum(prob * prob)
    return gini_indexx


def information_gain(Y: pd.Series, attr: pd.Series, criterion: str) -> float:
    """
    Function to calculate the information gain using criterion (entropy, gini index or MSE)
    """

    if criterion == "entropy":
        parent_impurity = entropy(Y)
    elif criterion == "gini_index":
        parent_impurity = gini_index(Y)
    elif criterion == "mse":
        parent_impurity = np.var(Y)
    else:
        raise ValueError(f"{criterion} not supported. Try among entropy, gini_index or mse.")

    unique_values = attr.unique()
    
    weighted_child_impurity = 0.0
    total_samples = len(Y)
    
    for value in unique_values:
        child_y = Y[attr == value]
        
        if len(child_y) == 0:
            continue
            
        weight = len(child_y) / total_samples
        
        if criterion == "entropy":
            child_impurity = entropy(child_y)
        elif criterion == "gini_index":
            child_impurity = gini_index(child_y)
        elif criterion == "mse":
            child_impurity = np.var(child_y)
        
        weighted_child_impurity += weight * child_impurity
    
    info_gain = parent_impurity - weighted_child_impurity
    return info_gain


def opt_split_attribute(X: pd.DataFrame, y: pd.Series, criterion, features: pd.Series):
    """
    Function to find the optimal attribute to split about.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    features: pd.Series is a list of all the attributes we have to split upon

    return: attribute to split upon
    """

    # According to wheather the features are real or discrete valued and the criterion, find the attribute from the features series with the maximum information gain (entropy or varinace based on the type of output) or minimum gini index (discrete output).

    best_attr = None
    best_score = -np.inf if criterion in ["entropy", "mse"] else np.inf

    for attr in features:
        score = information_gain(y, X[attr], criterion)
        if(criterion in ["entropy", "mse"] and score > best_score):
            best_score = score
            best_attr = attr
        elif(criterion == "gini_index" and score < best_score):
            best_score = score
            best_attr = attr

    return best_attr


def split_data(X: pd.DataFrame, y: pd.Series, attribute, value):
    """
    Funtion to split the data according to an attribute.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    attribute: attribute/feature to split upon
    value: value of that attribute to split upon

    return: splitted data(Input and output)
    """

    # Split the data based on a particular value of a particular attribute. You may use masking as a tool to split the data.

    

    pass
