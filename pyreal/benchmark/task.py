import os
import pickle

from sklearn.preprocessing import LabelEncoder

from pyreal.utils.transformer import (
    MultiTypeImputer, OneHotEncoderWrapper, fit_transforms, run_transforms,)


class Task:
    def __init__(self, X, y, model, transforms, name):
        self.X = X
        self.y = y
        self.model = model
        self.transforms = transforms
        self.name = name


def create_task(df, dataset_name, model_func):
    y = df["target"]
    X = df.drop("target", axis="columns")

    transforms = [MultiTypeImputer(), OneHotEncoderWrapper(
        feature_list=X.select_dtypes(include=["object", "category"]).columns)]

    fit_transforms(transforms, X)
    Xt = run_transforms(transforms, X)

    y = LabelEncoder().fit_transform(y)

    task_name = dataset_name + "_" + model_func.__name__

    filename = os.path.join("models", task_name + ".pkl")
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            model = pickle.load(f)
    else:
        model = model_func(Xt, y)
        with open(filename, "wb") as f:
            pickle.dump(model, f)

    return Task(X, y, model, transforms, task_name)
