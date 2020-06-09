"""
Contains helper functions for reading in and parsing the family screening
dataset and model
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from data_processing.feature_types import \
    BooleanFeature, IntNumericFeature, CategoricalFeature
from utils import synthetic_data_gen


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


def convert_to_categorical(X, mappings, defaults):
    cols = X.columns
    num_rows = X.shape[0]
    cat_cols = mappings['Feature'].values
    default_cats = defaults['Feature'].values
    cat_data = {}
    for col in cols:
        if col not in cat_cols:
            cat_data[col] = X[col]
        if col in cat_cols:
            new_name = mappings[mappings['Feature'] == col]["InterpretableFeature"].values[0]
            if new_name not in cat_data and new_name not in default_cats:
                cat_data[new_name] = np.empty(num_rows, dtype='object')
            elif new_name not in cat_data:
                cat_data[new_name] = np.full(num_rows, defaults[defaults['Feature'] == new_name]["Value"])
            cat_data[new_name][np.where(X[col]==1)] = mappings[mappings['Feature'] == col]["Value"].values[0]
    return pd.DataFrame(cat_data)


def convert_from_categorical(cat_data, mappings):
    cols = cat_data.columns
    num_rows = cat_data.shape[0]

    # List of column names that will be converted to one-hot
    cat_cols = mappings['InterpretableFeature'].values

    data = {}
    for col in cols:
        if col not in cat_cols:
            data[col] = cat_data[col]
        if col in cat_cols:
            values = cat_data[col]
            relevant_rows = mappings[mappings['InterpretableFeature'] == col]
            for ind in relevant_rows.index:
                new_col_name = relevant_rows['Feature'][ind]
                data[new_col_name] = np.zeros(num_rows)
                data[new_col_name][np.where(values == relevant_rows['Value'][ind])] = 1
    return pd.DataFrame(data)


def generate_data(X_cat, N, save_file=None):
    return synthetic_data_gen.generate_data(X_cat, N,
                                            return_result=True,
                                            save_file=save_file)


def to_probs(arr):
    odds = np.exp(arr)
    return odds / (1 + odds)


def preds_to_scores(preds, thresholds):
    return np.digitize(to_probs(preds), thresholds, right=True)


def predict_class(x, model, thresholds):
    pred = model.predict(x)
    scores = preds_to_scores(pred, thresholds)
    return scores
