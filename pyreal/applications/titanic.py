import os
import pickle

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def load_titanic_data():
    x_orig = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    y = x_orig["Survived"]
    x_orig = x_orig.drop("Survived", axis="columns")

    return x_orig, y


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


def load_titanic_model():
    return pickle.load(open(os.path.join(DATA_DIR, "model.pkl"), "rb"))


def load_titanic_transformers():
    return pickle.load(open(os.path.join(DATA_DIR, "transformers.pkl"), "rb"))
