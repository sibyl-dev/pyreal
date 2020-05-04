from examples.family_screening import family_screening_helpers as helpers

model = helpers.load_model()
thresholds = helpers.load_thresholds()

features = helpers.get_features()
feature_types = helpers.get_feature_types(features)

X, y = helpers.load_data(features)


def predict_class(x):
    pred = model.predict(x)
    scores = helpers.preds_to_scores(pred, thresholds)
    return scores
