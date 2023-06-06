import os
import pickle
from urllib.parse import urljoin

import pandas as pd
from lightgbm import LGBMClassifier

from pyreal import RealApp
from pyreal.transformers import (
    ColumnDropTransformer,
    MultiTypeImputer,
    OneHotEncoder,
    fit_transformers,
    run_transformers,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data_titanic")
DATA_FILE = os.path.join(DATA_DIR, "data.csv")
MODEL_FILE = os.path.join(DATA_DIR, "model.pkl")
TRANSFORMER_FILE = os.path.join(DATA_DIR, "transformers.pkl")
AWS_BASE_URL = "https://pyreal-data.s3.amazonaws.com/"


def load_data(n_rows=None, include_targets=False):
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
    else:
        url = urljoin(AWS_BASE_URL, "titanic.csv")
        df = pd.read_csv(url)

        if not os.path.isdir(DATA_DIR):
            os.mkdir(DATA_DIR)
        df.to_csv(DATA_FILE, index=False)
    y = df["target"].rename("Survived")
    x_orig = df.drop("target", axis="columns")

    if n_rows is not None and include_targets:
        return x_orig[:n_rows], y[:n_rows]
    elif n_rows is not None:
        return x_orig[:n_rows]
    elif include_targets:
        return x_orig, y
    else:
        return x_orig


def load_feature_descriptions():
    return {
        "PassengerId": "Passenger ID",
        "Pclass": "Ticket Class",
        "SibSp": "Number of siblings/spouses aboard",
        "Parch": "Number of parents/children aboard",
        "Ticket": "Ticket Number",
        "Fare": "Passenger Fare",
        "Cabin": "Cabin Number",
        "Embarked": "Port of Embarkment",
    }


def load_model():
    if os.path.exists(MODEL_FILE):
        return pickle.load(open(os.path.join(DATA_DIR, "model.pkl"), "rb"))
    else:
        transformers = load_transformers()
        x_orig, y = load_data(include_targets=True)
        x_model = run_transformers(transformers, x_orig)
        model = LGBMClassifier()
        model.fit(x_model, y)

        if not os.path.isdir(DATA_DIR):
            os.mkdir(DATA_DIR)
        with open(MODEL_FILE, "wb") as f:
            pickle.dump(model, f)
        return model


def load_transformers():
    if os.path.exists(TRANSFORMER_FILE):
        return pickle.load(open(os.path.join(DATA_DIR, "transformers.pkl"), "rb"))
    else:
        x_orig, y = load_data(include_targets=True)
        column_drop = ColumnDropTransformer(["PassengerId", "Name", "Ticket", "Cabin"])
        imputer = MultiTypeImputer()
        one_hot_encoder = OneHotEncoder(["Sex", "Embarked"])

        transformers = [column_drop, imputer, one_hot_encoder]
        fit_transformers(transformers, x_orig)

        if not os.path.isdir(DATA_DIR):
            os.mkdir(DATA_DIR)
        with open(TRANSFORMER_FILE, "wb") as f:
            pickle.dump(transformers, f)
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
    )
