from typing import Union
import pandas as pd
import numpy as np


def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the accuracy
    """

    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    assert y_hat.size == y.size
    # TODO: Write here

    correct_predictions = (y_hat == y).sum()
    accuracy = correct_predictions / y.size
    return accuracy


def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the precision
    """

    assert y_hat.size == y.size
    # TODO: Write here

    true_positives = ((y_hat == cls) & (y == cls)).sum()
    false_positives = ((y_hat == cls) & (y != cls)).sum()
    precision = true_positives / (true_positives + false_positives + 1e-9)
    return precision


def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the recall
    """

    assert y_hat.size == y.size
    # TODO: Write here

    true_positives = ((y_hat == cls) & (y == cls)).sum()
    false_negatives = ((y_hat != cls) & (y == cls)).sum()
    recall = true_positives / (true_positives + false_negatives + 1e-9)
    return recall


def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the root-mean-squared-error(rmse)
    """

    assert y_hat.size == y.size
    # TODO: Write here

    squared_errors = (y_hat - y) ** 2
    rmse = np.sqrt(squared_errors.mean())
    return rmse


def mae(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the mean-absolute-error(mae)
    """

    assert y_hat.size == y.size
    # TODO: Write here

    absolute_errors = (y_hat - y).abs()
    mae = absolute_errors.mean()
    return mae
