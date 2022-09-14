import numpy as np
import pandas as pd
import pytest

from pyreal.explainers.time_series.saliency.univariate_lime_saliency import UnivariateLimeSaliency

class ModelDummy:
    """
    Test class that takes in variable length inputs
    """

    def predict(self, x):
        print(np.array([1/np.sum(x, axis=1), 1-(1/np.sum(x, axis=1))]).shape)
        return np.array([1/np.sum(x, axis=0), 1-(1/np.sum(x, axis=0))])


def test_produce_lime_classification_no_transforms(classification_no_transforms):
    model = ModelDummy()
    explainer = UnivariateLimeSaliency(
        model=model,
        x_train_orig=pd.DataFrame([[1, 0, 0], [2, 0, 2], [3, 3, 0]]),
        y_orig=pd.DataFrame([[1, 0], [0, 1], [0, 1]]),
        transformers=[],
        fit_on_init=True,
    )

    x_one_dim = pd.DataFrame([[1, 1, 1]], columns=["A", "B", "C"])

    contributions = explainer.produce(x_one_dim)[0]
    print(contributions)

    assert contributions.shape == (3, 3)
    assert contributions["A"][0] == 0
    assert contributions["B"][0] == 2
    assert contributions["C"][0] == 0
