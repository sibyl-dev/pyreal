import os
import pickle

from sklearn.preprocessing import LabelEncoder

from pyreal.utils.transformer import (
    MultiTypeImputer, OneHotEncoderWrapper, fit_transforms, run_transforms,)


class Dataset:
    def __init__(self, X, y, model, transforms, name):
        self.X = X
        self.y = y
        self.model = model
        self.transforms = transforms
        self.name = name


def create_dataset(dataset_obj, model_func):
    X, y, categorical_indicator, attribute_names = dataset_obj.get_data(
        target=dataset_obj.default_target_attribute, dataset_format="dataframe")

    transforms = [MultiTypeImputer(), OneHotEncoderWrapper(
        feature_list=X.select_dtypes(include=["object", "category"]).columns)]

    fit_transforms(transforms, X)
    Xt = run_transforms(transforms, X)

    y = LabelEncoder().fit_transform(y)

    name = str(dataset_obj.name) + "_" + model_func.__name__

    filename = os.path.join("models", name + ".pkl")
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            model = pickle.load(f)
    else:
        model = model_func(Xt, y)
        with open(filename, "wb") as f:
            pickle.dump(model, f)

    return Dataset(X, y, model, transforms, name)
