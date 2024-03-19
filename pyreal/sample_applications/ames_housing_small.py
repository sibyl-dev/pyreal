import os

import pandas as pd
from lightgbm import LGBMRegressor
from pyreal import RealApp
from pyreal.transformers import (
    OneHotEncoder,
    MultiTypeImputer,
    StandardScaler,
    fit_transformers,
    run_transformers,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data_ames_housing_small")
INPUT_DATA_FILE = os.path.join(DATA_DIR, "input_data.csv")
DATA_FILE = os.path.join(DATA_DIR, "data.csv")
MODEL_FILE = os.path.join(DATA_DIR, "model.pkl")
DESCRIPTION_FILE = os.path.join(DATA_DIR, "feature_descriptions.csv")
TRANSFORMER_FILE = os.path.join(DATA_DIR, "transformers.pkl")


def load_feature_descriptions():
    descriptions_df = pd.read_csv(DESCRIPTION_FILE)
    return descriptions_df.set_index("Name").to_dict()["Description"]


def load_data(n_rows=None, include_targets=False):
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
    else:
        raise FileNotFoundError("Ames housing data is missing")

    df = df[df["HouseSize"] < 4000]
    y = df["SalePrice"]
    x_orig = df.drop("SalePrice", axis="columns")
    x_orig["CentralAir"].replace({"True": True, "False": False}, inplace=True)

    if n_rows is not None and include_targets:
        return x_orig[:n_rows], y[:n_rows]
    elif n_rows is not None:
        return x_orig[:n_rows]
    elif include_targets:
        return x_orig, y
    else:
        return x_orig


def load_input_data():
    if os.path.exists(INPUT_DATA_FILE):
        x_orig = pd.read_csv(INPUT_DATA_FILE)
    else:
        raise FileNotFoundError("Ames housing data is missing")

    x_orig["CentralAir"].replace({"True": True, "False": False}, inplace=True)

    return x_orig


def load_model():
    transformers = load_transformers()
    x_orig, y = load_data(include_targets=True)
    x_orig = x_orig.drop("House ID", axis="columns")
    x_model = run_transformers(transformers, x_orig)
    model = LGBMRegressor()
    model.fit(x_model, y)

    return model


def load_transformers():
    x_orig = load_data()
    x_orig = x_orig.drop("House ID", axis="columns")
    imputer = MultiTypeImputer()
    onehotencoder = OneHotEncoder(columns=["Neighborhood", "Exterior1st"])
    scaler = StandardScaler()
    transformers = [imputer, onehotencoder, scaler]
    fit_transformers(transformers, x_orig)
    return transformers


def load_app():
    x_train_orig, y = load_data(include_targets=True)
    model = load_model()
    transformers = load_transformers()
    feature_descriptions = load_feature_descriptions()

    return RealApp(
        model,
        x_train_orig,
        y_train=y,
        transformers=transformers,
        feature_descriptions=feature_descriptions,
        id_column="House ID",
    )
