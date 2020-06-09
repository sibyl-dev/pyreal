from examples.family_screening import family_screening_helpers as helpers
from utils.find_representative_data import kmedoids
import pickle
import pandas as pd
import os

from sklearn import manifold, datasets
import time
import matplotlib.pyplot as plt
from functools import partial
import numpy as np


def predict_class(x):
    pred = model.predict(x)
    scores = helpers.preds_to_scores(pred, thresholds)
    return scores


# TODO: generalize this/make configurable
directory = "../../../../Family Screening Project/data_and_models/"
dataset_filename = os.path.join(
    directory, "Model_Dataset_Version_01_01_partition_predictions.csv")
features_filename = os.path.join(
    directory, "Design_Specification_DOUGLAS_v1.4.csv")
model_filename = os.path.join(directory, "weights_model_feb2019.csv")
threshold_filename = os.path.join(directory, "Performance_Matrix.csv")
feature_mappings_filename = os.path.join(
    directory, "interpretable_features.csv")
defaults_filename = os.path.join(directory, "default_values.csv")

# Load model
model = helpers.load_model(model_filename)
thresholds = helpers.load_thresholds(threshold_filename)

# Load data
features = helpers.get_features(model_filename)
feature_types = helpers.get_feature_types(features, features_filename)
X, y = helpers.load_data(features, dataset_filename)
y_preds = predict_class(X)

# Load info about interpretable features
mappings, defaults = helpers.load_feature_mappings(
    feature_mappings_filename, defaults_filename)

# Convert to categorical features
X_cat = helpers.convert_to_categorical(X, mappings, defaults)

# Generate fake data
X_cat_synth = helpers.generate_data(X_cat, 100, "synth_data.csv")
X_cat = helpers.convert_from_categorical(X_cat_synth, mappings)
synth_preds = predict_class(X_cat)
print(synth_preds)


X_sample = X.sample(1000)
y_sample = y_preds[X_sample.index]
prototypes = kmedoids(X_sample, 50, 2, max_steps=10)[0]
prototype_inds = X.index[prototypes]
