import numpy as np
import pandas as pd

from pyreal.explainers import UnivariateLimeSaliency


class ModelDummyClass:
    def predict(self, x):
        return np.squeeze(np.array([1 / x[:, 0], 1 / x[:, 1], 1 / x[:, 2]])).T


class ModelDummyReg:
    def predict(self, x):
        return x[:, 0]


def test_produce_lime_classification_no_transforms(classification_no_transforms):
    model = ModelDummyClass()
    explainer = UnivariateLimeSaliency(
        model=model,
        x_train_orig=pd.DataFrame([[3, 2, 1], [1, 2, 3], [6, 2, 5]]),
        y_train=pd.DataFrame([[0, 0, 1], [1, 0, 0], [0, 1, 0]]),
        transformers=[],
        regression=False,
        fit_on_init=True,
        classes=[0, 1, 2],
    )

    x_one_dim = pd.DataFrame([[1, 1, 1]], columns=["A", "B", "C"])

    contributions = explainer.produce(x_one_dim).get()
    assert contributions.shape == (3, 3)
    assert contributions["A"][0] > 0.1
    assert contributions["B"][1] > 0.1
    assert contributions["C"][2] > 0.1

    assert (np.abs(contributions["A"].iloc[1:2]) < 0.01).all()
    assert np.abs(contributions["B"].iloc[0]) < 0.01
    assert np.abs(contributions["B"].iloc[2]) < 0.01
    assert (np.abs(contributions["C"].iloc[0:1]) < 0.01).all()


def test_produce_lime_regression_no_transforms(regression_no_transforms):
    model = ModelDummyReg()
    explainer = UnivariateLimeSaliency(
        model=model,
        x_train_orig=pd.DataFrame([[1, 0, 0], [2, 0, 2], [3, 3, 0]]),
        y_train=pd.Series([1, 2, 3]),
        transformers=[],
        regression=True,
        fit_on_init=True,
    )

    x_one_dim = pd.DataFrame([[1, 1, 1]], columns=["A", "B", "C"])

    contributions = explainer.produce(x_one_dim).get()

    assert contributions.shape == (1, 3)
    assert contributions["A"].iloc[0] < -0.01
    assert (np.abs(contributions["B"]) < 0.001).all()
    assert (np.abs(contributions["C"]) < 0.001).all()
