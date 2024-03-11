from pyreal.sample_applications import ames_housing
from pyreal.explainers import LocalFeatureContribution

X_train, y_train = ames_housing.load_data(include_targets=True)
X_train = X_train.drop(columns=["Id"])
explainer = LocalFeatureContribution(
    model=ames_housing.load_model(),
    x_train_orig=X_train,
    y_train=y_train,
    transformers=ames_housing.load_transformers(),
    feature_descriptions=ames_housing.load_feature_descriptions(),
)
explainer.fit()
print("starting")
explainer.produce_narrative_explanation(X_train.iloc[0:5], num_features=2)
