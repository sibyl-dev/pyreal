import os
import pickle

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler

from pyreal import RealApp
from pyreal.transformers import (
    DataFrameWrapper,
    OneHotEncoder,
    Transformer,
    fit_transformers,
    run_transformers,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data_student")
DATA_FILE = os.path.join(DATA_DIR, "data.csv")
STUDENTS_FILE = os.path.join(DATA_DIR, "students.csv")
MODEL_FILE = os.path.join(DATA_DIR, "model.pkl")
TRANSFORMER_FILE = os.path.join(DATA_DIR, "transformers.pkl")


class BooleanEncoder(Transformer):
    def __init__(self, cols, **kwargs):
        self.cols = cols
        super().__init__(**kwargs)

    def data_transform(self, x):
        x_transform = x.copy()
        for col in self.cols:
            x_transform[col] = x_transform[col].replace(("yes", "no"), (1, 0))
        x_transform["famsize"] = x_transform["famsize"].astype("category")
        x_transform["famsize"] = x_transform["famsize"].cat.set_categories(["LE3", "GT3"])
        x_transform["famsize"] = x_transform["famsize"].cat.reorder_categories(["LE3", "GT3"])
        x_transform["famsize"] = x_transform["famsize"].cat.codes
        return x_transform


def load_feature_descriptions():
    return {
        "school": "School",
        "sex": "Sex",
        "age": "Age",
        "address": "Address type",
        "famsize": "Family size",
        "Pstatus": "Parent's cohibition status",
        "Medu": "Mother's education",
        "Fedu": "Father's education",
        "Mjob": "Mother's job",
        "Fjob": "Father's job",
        "reason": "Reason for choosing this school",
        "guardian": "Student's guardian",
        "traveltime": "Home to school travel time",
        "studytime": "Weekly study time",
        "failures": "Number of past class failures",
        "schoolsup": "Extra education support",
        "famsup": "Family eductional support",
        "paid": "Extra paid classes within the subject",
        "activities": "Extra-curricular activities",
        "nursery": "Attended nursery school",
        "higher": "Wants to take higher education",
        "internet": "Has internet at home",
        "romantic": "In a romantic relationship",
        "famrel": "Quality of family relationships (1-5)",
        "freetime": "Amount of free time after school (1-5)",
        "goout": "Frequency of going out with friends (1-5)",
        "Dalc": "Frequency of workday alcohol consumption (1-5)",
        "Walc": "Frequency of weekend alcohol consumption (1-5)",
        "health": "Current health status (1-5)",
        "absences": "Number of school absences",
    }


def load_data(n_rows=None, filename=None):
    if filename is None:
        filename = DATA_FILE
    df = pd.read_csv(filename)
    y = (df["G3"] > 10).astype(int)

    data = df.drop(["G1", "G2", "G3"], axis="columns")
    x_orig = data

    if n_rows is not None:
        return x_orig[:n_rows], y[:n_rows]
    return x_orig, y


def load_students():
    x, _ = load_data(filename=STUDENTS_FILE)
    return x


def load_model():
    if os.path.exists(MODEL_FILE):
        return pickle.load(open(os.path.join(DATA_DIR, "model.pkl"), "rb"))
    else:
        transformers = load_transformers()
        x_orig, y = load_data()
        x_model = run_transformers(transformers, x_orig)
        model = LGBMClassifier()
        model.fit(x_model, y)

        if not os.path.isdir(DATA_DIR):
            os.mkdir(DATA_DIR)
        with open(MODEL_FILE, "wb") as f:
            pickle.dump(model, f)
        return model


def load_transformers(x=None):
    if x is None:
        x, _ = load_data()
    onehotencoder = OneHotEncoder(
        ["school", "sex", "address", "Pstatus", "reason", "guardian", "Mjob", "Fjob"], model=True
    )
    boolean_encoder = BooleanEncoder(
        ["schoolsup", "famsup", "paid", "activities", "nursery", "internet", "romantic", "higher"],
        model=True,
    )
    standard_scaler = DataFrameWrapper(StandardScaler(), model=True)

    transformers = [onehotencoder, boolean_encoder, standard_scaler]
    if x is not None:
        fit_transformers(transformers, x)
    return transformers


def load_app():
    x_train_orig, y = load_data()
    model = load_model()
    transformers = load_transformers()
    feature_descriptions = load_feature_descriptions()

    return RealApp(
        model,
        x_train_orig,
        y_train=y,
        transformers=transformers,
        feature_descriptions=feature_descriptions,
        id_column="name",
    )
