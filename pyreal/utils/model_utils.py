"""
Provides helper functions for more flexibility with Model inputs
"""

import pickle

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression


def load_model_from_pickle(filepath):
    """
    Load the model from a pickle filepath

    Args:
        filepath (string filepath):
            The location of the pickled mode

    Returns:
        model object:
            The loaded model

    """
    with open(filepath, "rb") as f:
        return pickle.load(f)


def load_model_from_weights(weights, model_type, includes_intercept=True):
    """
    Generates an sklearn model from a list of weights

    Args:
        weights (array_like):
           Ordered list of model weights. Can be a list/numpy array, in which feature
        model_type (string):
            Base model type. One of: `linear_regression` and `logistic_regression`
        includes_intercept: Boolean
           True if first element of weights is the intercept, False otherwise
           If False, intercept defaults to 0

    Returns:
        sklearn model
            The loaded model with given weights
    """
    if model_type == "linear_regression":
        model = LinearRegression()
    elif model_type == "logistic_regression":
        model = LogisticRegression()
    else:
        raise ValueError("Unrecognized model type %s" % model_type)

    weights = np.asanyarray(weights)
    if includes_intercept:
        weight_size = weights.size - 1
    else:
        weight_size = weights.size
    dummy_X = np.zeros((2, weight_size))
    dummy_y = np.zeros(2)
    model.fit(dummy_X, dummy_y)

    if includes_intercept:
        model.coef_ = np.array(weights[1:])
        model.intercept_ = weights[0]
    else:
        model.coef_ = np.array(weights)
        model.intercept_ = 0
    return model
