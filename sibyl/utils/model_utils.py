"""
Provides helper functions for more flexibility with Model inputs
"""

from sklearn.linear_model import LinearRegression, LogisticRegression
import numpy as np


def load_model_from_weights(weights, model_type):
    """
    Generates an sklearn model from a list of weights
    :param weights: array_like
           Ordered list of model weights. Can be a list/numpy array, in which feature
    :param feature_names:
    :param model_type:
    :return:
    """
    if model_type == "linear_regression":
        model = LinearRegression()
    elif model_type == "logistic_regression":
        model = LogisticRegression()
    else:
        raise ValueError("Unrecognized model type %s" % model_type)

    weights = np.asanyarray(weights)
    dummy_X = np.zeros((1, weights.shape[1] - 1))
    dummy_y = np.zeros(weights.shape[1] - 1)
    model.fit(dummy_X, dummy_y)

    model.coef_ = np.array(weights["weight"][1:])
    model.intercept_ = weights["weight"][0]
    return model

