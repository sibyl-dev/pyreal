"""
Contains helper functions for reading in and parsing the family screening
dataset and model
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from data_processing.feature_types import \
    BooleanFeature, IntNumericFeature, CategoricalFeature


def load_model(model_filepath):
    """
    Load the model
    :return: the model
    """
    model_weights = pd.read_csv(model_filepath)

    model = Lasso()
    dummy_X = np.zeros((1, model_weights.shape[1]-1))
    dummy_y = np.zeros(model_weights.shape[1]-1)
    model.fit(dummy_X, dummy_y)

    model.coef_ = np.array(model_weights["weight"][1:])
    model.intercept_ = model_weights["weight"][0]
    return model


def get_features(model_filepath):
    model_weights = pd.read_csv(model_filepath)
    return model_weights["name"].drop(0)


def load_thresholds(directory_filepath):
    """
    Load the score bin thresholds
    :return: list of thresholds
    """
    thresholds = pd.read_csv(directory_filepath)
    thresholds = thresholds["Upper Bound"].to_numpy()[::-1]
    return thresholds


def load_data(features, dataset_filepath):
    data = pd.read_csv(dataset_filepath)

    y = data.PRO_PLSM_NEXT730_DUMMY
    X = data[features]

    return X, y


def load_readable_data(X, y):
    """
    Load the data and convert to a human-readable form
    :return X_readable: array_like of size (n_samples, n_features)
            y_readable: array_like of size (n_samples,)
    """
    #mappings = load_feature_mappings()

    return X, y


def transform_data(X_readable, y_readable):
    """
    Transform human-readable data to model-input data

    :param X_readable:
    :param y_readable:
    :return:
    """
    X = None
    y = None
    return X, y


def get_feature_types(features, features_filepath):
    feature_descriptions = pd.read_csv(features_filepath,
                                       header=0,encoding='latin-1')
    feature_descriptions = feature_descriptions[
        feature_descriptions["Name"].isin(features)]
    raw_types = feature_descriptions["Variable Type"].reset_index(drop=True)
    types = pd.Series([BooleanFeature()]*raw_types.size)

    types[raw_types == "categorical [0,1]"] = BooleanFeature()
    types[raw_types == "numeric/age"] = IntNumericFeature(0, 99)
    types[raw_types == "numeric/days in placement"] = IntNumericFeature(0, 365)
    types[raw_types.str.contains("numeric")] = IntNumericFeature(0, 300)

    return types


def load_feature_mappings(mappings_filepath, defaults_filepath):
    """
    Load in the categorical data to one hot feature mappings
    :return: 2 columns (feature -> combined feature),
             2 columns (combined feature -> default)
    """
    feature_mappings = pd.read_csv(mappings_filepath,
                                       header=0, encoding='latin-1')
    defaults = pd.read_csv(defaults_filepath)
    return feature_mappings.dropna(axis='index'), defaults.dropna(axis='index')


def to_probs(arr):
    odds = np.exp(arr)
    return odds / (1 + odds)


def preds_to_scores(preds, thresholds):
    return np.digitize(to_probs(preds), thresholds, right=True)


def predict_class(x, model, thresholds):
    pred = model.predict(x)
    scores = preds_to_scores(pred, thresholds)
    return scores
