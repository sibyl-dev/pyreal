"""
Provides helper functions for more flexibility with Model inputs
"""

from sklearn.linear_model import LinearRegression, LogisticRegression
import numpy as np
import pickle


def load_model_from_pickle(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)


def load_model_from_weights(weights, model_type, includes_intercept=True):
    """
    Generates an sklearn model from a list of weights
    :param weights: array_like
           Ordered list of model weights. Can be a list/numpy array, in which feature
    :param feature_names:
    :param model_type:
    :param includes_intercept: Boolean
           True if first element of weights is the intercept, False otherwise
           If False, intercept defaults to 0
    :return:
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
    print(dummy_X.shape, weights[1:].shape)
    model.fit(dummy_X, dummy_y)

    if includes_intercept:
        model.coef_ = np.array(weights[1:])
        model.intercept_ = weights[0]
    else:
        model.coef_ = np.array(weights)
        model.intercept_ = 0
    return model

