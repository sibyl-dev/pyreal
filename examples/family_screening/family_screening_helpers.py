"""
Contains helper functions for reading in and parsing the family screening
dataset and model
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from data_processing.feature_types import \
    BooleanFeature, IntNumericFeature, CategoricalFeature

directory = "../../../../Family Screening Project/data_and_models/"
dataset_filename = "Model_Dataset_Version_01_01_partition_predictions.csv"
features_filename = "Design_Specification_DOUGLAS_v1.4.csv"
model_filename = "weights_model_feb2019.csv"
threshold_filename = "Performance_Matrix.csv"
feature_mappings_filename = "interpretable_features.csv"
defaults_filename = "default_values.csv"


def load_model():
    """
    Load the model
    :return: the model
    """
    model_weights = pd.read_csv(directory+model_filename)

    model = Lasso()
    dummy_X = np.zeros((1, model_weights.shape[1]-1))
    dummy_y = np.zeros(model_weights.shape[1]-1)
    model.fit(dummy_X, dummy_y)

    model.coef_ = np.array(model_weights["weight"][1:])
    model.intercept_ = model_weights["weight"][0]
    return model


def get_features():
    model_weights = pd.read_csv(directory + model_filename)
    return model_weights["name"].drop(0)


def load_thresholds():
    """
    Load the score bin thresholds
    :return: list of thresholds
    """
    thresholds = pd.read_csv(directory+threshold_filename)
    thresholds = thresholds["Upper Bound"].to_numpy()[::-1]
    return thresholds


def load_data(features):
    """
    Load the data
    :return X: array_like of size (n_samples, n_features)
            y: array_like of size (n_samples,)
    """
    data = pd.read_csv(directory+dataset_filename)

    y = data.PRO_PLSM_NEXT730_DUMMY
    X = data[features]

    return X, y


def get_feature_types(features):
    feature_descriptions = pd.read_csv(directory+features_filename,
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


def load_feature_mappings():
    feature_mappings = pd.read_csv(directory + feature_mappings_filename,
                                       header=0, encoding='latin-1')
    defaults = pd.read_csv(directory+defaults_filename)
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
