from examples.housing import housing_helpers as helpers

model = helpers.load_model()

X_orig, y_orig = helpers.load_data()
X, y = helpers.transform_for_model(X_orig, y_orig)
