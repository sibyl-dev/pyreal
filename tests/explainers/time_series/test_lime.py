import numpy as np
import pandas as pd
import pytest

from pyreal.explainers.time_series.saliency.univariate_lime_saliency import UnivariateLimeSaliency

class ModelDummyClass:
    def predict(self, x):
        return np.squeeze(np.array([1/x[:, 0], 1-(1/x[:, 0])])).T


class ModelDummyReg:
    def predict(self, x):
        return 1/x[:, 0]


def test_produce_lime_classification_no_transforms(classification_no_transforms):
    model = ModelDummyClass()
    explainer = UnivariateLimeSaliency(
        model=model,
        x_train_orig=pd.DataFrame([[1, 0, 0], [2, 0, 2], [3, 3, 0]]),
        y_orig=pd.DataFrame([[1, 0], [0, 1], [0, 1]]),
        transformers=[],
        regression=True,
        fit_on_init=True
    )

    x_one_dim = pd.DataFrame([[1, 1, 1]], columns=["A", "B", "C"])
    print(model.predict(np.array(x_one_dim)))

    contributions = explainer.produce(x_one_dim)[0]

    assert contributions.shape == (2, 3)
    assert contributions["A"][0] < -.01
    assert contributions["A"][1] > .01
    assert (np.abs(contributions["B"]) < .001).all()
    assert (np.abs(contributions["C"]) < .001).all()


def test_produce_lime_regression_no_transforms(regression_no_transforms):
    model = ModelDummyReg()
    explainer = UnivariateLimeSaliency(
        model=model,
        x_train_orig=pd.DataFrame([[1, 0, 0], [2, 0, 2], [3, 3, 0]]),
        y_orig=pd.DataFrame([1, .5, .33]),
        transformers=[],
        regression=True,
        fit_on_init=True
    )

    x_one_dim = pd.DataFrame([[1, 1, 1]], columns=["A", "B", "C"])
    print(model.predict(x_one_dim.numpy()))

    contributions = explainer.produce(x_one_dim)[0]
    print(contributions)

    assert contributions.shape == (1, 3)
    assert contributions["A"] < -.01
    assert (np.abs(contributions["B"]) < .001).all()
    assert (np.abs(contributions["C"]) < .001).all()

