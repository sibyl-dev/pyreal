import os
import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from pyreal.transformers.one_hot_encode import OneHotEncoder


class DummyModel:
    def __init__(self, value):
        self.value = value

    def fit(self, x):
        return self

    def predict(self, x):
        return np.sum(x, axis=1) + self.value


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
def dummy_models():
    return {"id0": DummyModel(0), "id1": DummyModel(1), "id2": DummyModel(2)}


@pytest.fixture()
def dummy_model():
    return DummyModel(0)


@pytest.fixture()
def transformer_test_data():
    x = pd.DataFrame([[2, 1, 3, 9], [4, 3, 4, 0], [6, 7, 2, 2]], columns=["A", "B", "C", "D"])
    columns = ["C", "A"]
    return {"x": x, "columns": columns}


@pytest.fixture()
def all_models(regression_no_transforms, regression_one_hot, classification_no_transforms):
    return [regression_no_transforms, regression_one_hot, classification_no_transforms]


@pytest.fixture()
def regression_no_transforms(test_root):
    x = pd.DataFrame([[2, 1, 3], [4, 3, 4], [6, 7, 2]], columns=["A", "B", "C"])
    y = x.iloc[:, 0:1].copy()
    model_no_transforms = LinearRegression()
    model_no_transforms.fit(x, y)
    model_no_transforms.coef_ = np.array([1, 0, 0])
    model_no_transforms.intercept_ = 0
    model_no_transforms_filename = os.path.join(test_root, "data", "model_no_transforms.pkl")
    with open(model_no_transforms_filename, "wb") as f:
        pickle.dump(model_no_transforms, f)

    return {"model": model_no_transforms_filename, "transformers": None, "x": x, "y": y}


@pytest.fixture()
def regression_no_transforms_big(test_root):
    data = np.stack((np.arange(100), np.arange(100) * -0.37, np.arange(100) * 0.2 + 3), axis=1)
    x = pd.DataFrame(data, columns=["A", "B", "C"])
    y = x.iloc[:, 0:1].copy()
    model_no_transforms = LinearRegression()
    model_no_transforms.fit(x, y)
    model_no_transforms.coef_ = np.array([1, 0, 0])
    model_no_transforms.intercept_ = 0
    model_no_transforms_filename = os.path.join(test_root, "data", "model_no_transforms.pkl")
    with open(model_no_transforms_filename, "wb") as f:
        pickle.dump(model_no_transforms, f)

    return {"model": model_no_transforms_filename, "transformers": None, "x": x, "y": y}


@pytest.fixture()
def classification_no_transforms(test_root):
    x = pd.DataFrame([[3, 0, 0], [0, 3, 0], [0, 0, 3]], columns=["A", "B", "C"])
    y = pd.Series([1, 1, 3])
    model_no_transforms = LogisticRegression()
    model_no_transforms.fit(x, pd.Series([1, 2, 3]))
    model_no_transforms.coef_ = np.array([[0, 1, 0], [0, 1, 0], [0, 0, 1]])
    model_no_transforms.intercept_ = np.array([0])
    model_no_transforms_filename = os.path.join(test_root, "data", "model_no_transforms.pkl")
    with open(model_no_transforms_filename, "wb") as f:
        pickle.dump(model_no_transforms, f)

    return {
        "model": model_no_transforms_filename,
        "transformers": None,
        "x": x,
        "y": y,
        "classes": np.arange(1, 4),
    }


@pytest.fixture()
def binary_classification_no_transforms(test_root):
    x = pd.DataFrame([[3, 0, 0], [0, 3, 0], [0, 0, 3]], columns=["A", "B", "C"])
    y = pd.Series([1, 1, 0])
    model_no_transforms = LogisticRegression()
    model_no_transforms.fit(x, pd.Series([1, 2, 3]))
    model_no_transforms.coef_ = np.array([[0, 1, 0], [0, 1, 0], [0, 0, 1]])
    model_no_transforms.intercept_ = np.array([0])
    model_no_transforms_filename = os.path.join(test_root, "data", "model_no_transforms.pkl")
    with open(model_no_transforms_filename, "wb") as f:
        pickle.dump(model_no_transforms, f)

    return {
        "model": model_no_transforms_filename,
        "transformers": None,
        "x": x,
        "y": y,
        "classes": np.arange(0, 2),
    }


@pytest.fixture()
def regression_one_hot(test_root):
    x = pd.DataFrame([[2, 1, 3], [4, 3, 4], [6, 7, 2]], columns=["A", "B", "C"])
    one_hot_encoder = OneHotEncoder(columns=["A"], model=True, interpret=False)
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
    x = pd.DataFrame(
        [[1, 1, 1], [2, 2.5, 3], [10, 11, 12], [11, 10.3, 10]], columns=["A", "B", "C"]
    )
    y = pd.DataFrame([0, 0, 1, 1])

    model_test_tree = LogisticRegression()
    model_test_tree.fit(x, y)

    model_no_transform_tree = os.path.join(test_root, "data", "model_no_transform_tree.pkl")
    with open(model_no_transform_tree, "wb") as f:
        pickle.dump(model_test_tree, f)
    return {"model": model_no_transform_tree, "transformers": None, "x": x, "y": y}


@pytest.fixture()
def time_series_data():
    n_inst, n_var, n_time = 4, 3, 10
    np3d = np.random.randn(n_inst, n_var, n_time)
    np2d = np.random.randn(n_inst, n_time)
    mi3d = pd.MultiIndex.from_product([[f"var_{i}" for i in range(n_var)], np.arange(n_time)])
    df3d = pd.DataFrame(data=np3d.reshape((4, 30)), columns=mi3d)
    mi2d = pd.MultiIndex.from_product([["var_0"], np.arange(n_time)])
    df2d = pd.DataFrame(data=np2d, columns=mi2d)

    nested = pd.DataFrame(columns=[f"var_{i}" for i in range(n_var)])
    for v in range(n_var):
        nested[f"var_{v}"] = [pd.Series(np3d[i, v, :]) for i in range(n_inst)]
    return {"np3d": np3d, "np2d": np2d, "df3d": df3d, "df2d": df2d, "nested": nested}
