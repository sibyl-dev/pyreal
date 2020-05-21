from examples.family_screening import family_screening_helpers as helpers

# TODO: generalize this/make configurable
directory = "../../../../Family Screening Project/data_and_models/"
dataset_filename = "Model_Dataset_Version_01_01_partition_predictions.csv"
features_filename = "Design_Specification_DOUGLAS_v1.4.csv"
model_filename = "weights_model_feb2019.csv"
threshold_filename = "Performance_Matrix.csv"
feature_mappings_filename = "interpretable_features.csv"
defaults_filename = "default_values.csv"

model = helpers.load_model(directory+model_filename)
thresholds = helpers.load_thresholds(directory+threshold_filename)

features = helpers.get_features(directory+model_filename)
feature_types = helpers.get_feature_types(features, directory+features_filename)

X, y = helpers.load_data(features, directory+dataset_filename)
#X, y = helpers.transform_data(X_readable, y_readable)


def predict_class(x):
    pred = model.predict(x)
    scores = helpers.preds_to_scores(pred, thresholds)
    return scores
