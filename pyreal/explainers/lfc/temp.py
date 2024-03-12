from pyreal.sample_applications import ames_housing
from pyreal.explainers import LocalFeatureContribution
from pyreal import RealApp
import yaml
import os

print("starting")
X_train, y_train = ames_housing.load_data(include_targets=True)
X_train = X_train.drop(columns=["Id"])
with open(os.path.join("..", "..", "..", "passwords.yml"), "r") as f:
    y = yaml.safe_load(f)
    openai_api_key = y["openai_api_key"]

app = RealApp(
    ames_housing.load_model(),
    X_train_orig=X_train,
    y_train=y_train,
    transformers=ames_housing.load_transformers(),
    openai_api_key=openai_api_key,
    fit_transformers=True,
    feature_descriptions=ames_housing.load_feature_descriptions(),
)

print(app.produce_narrative_feature_contributions(X_train.iloc[0], num_features=5))
