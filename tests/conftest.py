import os
import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from pyreal.transformers.one_hot_encode import OneHotEncoder


@pytest.fixture(scope="session", autouse=True)
def test_root():
    test_root = os.path.dirname(os.path.abspath(__file__))
    test_dir = os.path.join(test_root, "data")
    try:
        os.makedirs(test_dir)
    except FileExistsError:
        pass
    yield test_root
    for f in os.listdir(test_dir):
        os.remove(os.path.join(test_dir, f))


@pytest.fixture()
def all_models(regression_no_transforms, regression_one_hot, classification_no_transforms):
    return [regression_no_transforms, regression_one_hot, classification_no_transforms]


@pytest.fixture()
def regression_no_transforms(test_root):
    x = pd.DataFrame([[2, 1, 3],
                      [4, 3, 4],
                      [6, 7, 2]], columns=["A", "B", "C"])
    y = x.iloc[:, 0:1].copy()
    model_no_transforms = LinearRegression()
    model_no_transforms.fit(x, y)
    model_no_transforms.coef_ = np.array([1, 0, 0])
    model_no_transforms.intercept_ = 0
    model_no_transforms_filename = os.path.join(test_root, "data",
                                                "model_no_transforms.pkl")
    with open(model_no_transforms_filename, "wb") as f:
        pickle.dump(model_no_transforms, f)

    return {"model": model_no_transforms_filename, "transformers": None, "x": x, "y": y}


@pytest.fixture()
def classification_no_transforms(test_root):
    x = pd.DataFrame([[3, 0, 0],
                      [0, 3, 0],
                      [0, 0, 3]], columns=["A", "B", "C"])
    y = pd.Series([1, 1, 3])
    model_no_transforms = LogisticRegression()
    model_no_transforms.fit(x, pd.Series([1, 2, 3]))
    model_no_transforms.coef_ = np.array([[0, 1, 0],
                                          [0, 1, 0],
                                          [0, 0, 1]])
    model_no_transforms.intercept_ = np.array([0])
    model_no_transforms_filename = os.path.join(test_root, "data",
                                                "model_no_transforms.pkl")
    with open(model_no_transforms_filename, "wb") as f:
        pickle.dump(model_no_transforms, f)

    return {"model": model_no_transforms_filename, "transformers": None, "x": x, "y": y}


@pytest.fixture()
def regression_one_hot(test_root):
    x = pd.DataFrame([[2, 1, 3],
                      [4, 3, 4],
                      [6, 7, 2]], columns=["A", "B", "C"])
    one_hot_encoder = OneHotEncoder(columns=["A"])
    one_hot_encoder.fit(x)
    x_trans = one_hot_encoder.transform(x)
    y = pd.DataFrame([1, 2, 3])
    model_one_hot = LinearRegression()
    model_one_hot.fit(x_trans, y)
    model_one_hot.coef_ = np.array([0, 0, 1, 2, 3])
    model_one_hot.intercept_ = 0
    model_one_hot_filename = os.path.join(test_root, "data", "model_one_hot.pkl")
    with open(model_one_hot_filename, "wb") as f:
        pickle.dump(model_one_hot, f)
    return {"model": model_one_hot_filename, "transformers": one_hot_encoder, "x": x, "y": y}


@pytest.fixture()
def classification_no_transform_tree(test_root):
    x = pd.DataFrame([[1, 1, 1],
                      [2, 2.5, 3],
                      [10, 11, 12],
                      [11, 10.3, 10]], columns=["A", "B", "C"])
    y = pd.DataFrame([0, 0, 1, 1])

    model_test_tree = LogisticRegression()
    model_test_tree.fit(x, y)

    model_no_transform_tree = os.path.join(
        test_root, "data", "model_no_transform_tree.pkl")
    with open(model_no_transform_tree, "wb") as f:
        pickle.dump(model_test_tree, f)
    return {"model": model_no_transform_tree, "transformers": None, "x": x, "y": y}
