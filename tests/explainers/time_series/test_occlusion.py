import numpy as np
import pandas as pd
import pytest

from pyreal.explainers.time_series.saliency.univariate_occlusion_saliency import (
    UnivariateOcclusionSaliency,
)


def test_univariate_occlusion_no_transforms(regression_no_transforms):
    model = regression_no_transforms
    explainer = UnivariateOcclusionSaliency(
        model=model["model"],
        x_train_orig=model["x"],
        transformers=model["transformers"],
        width=1,
        k=0,
        fit_on_init=True,
        regression=True,
    )

    x_one_dim = pd.DataFrame([[2, 10, 10]], columns=["A", "B", "C"])
    contributions = explainer.produce(x_one_dim)[0]

    assert x_one_dim.shape == contributions.shape
    assert contributions.iloc[0, 0] == -2
    assert contributions.iloc[0, 1] == 0
    assert contributions.iloc[0, 2] == 0


def test_univariate_occlusion_multivariate_raise_error(regression_no_transforms):
    model = regression_no_transforms
    explainer = UnivariateOcclusionSaliency(
        model=model["model"],
        x_train_orig=model["x"],
        transformers=model["transformers"],
        width=1,
        fit_on_init=True,
        regression=True,
    )

    x_multi_dim = pd.DataFrame([[2, 1, 1], [4, 2, 3]], columns=["A", "B", "C"])

    with pytest.raises(ValueError):
        explainer.produce(x_multi_dim)


def test_produce_occlusion_classification_no_transforms(classification_no_transforms):
    model = classification_no_transforms
    explainer = UnivariateOcclusionSaliency(
        model=model["model"],
        x_train_orig=model["x"],
        width=1,
        k=0,
        transformers=model["transformers"],
        fit_on_init=True,
        classes=np.arange(1, 4),
    )

    x_one_dim = pd.DataFrame([[1, 1, 1]], columns=["A", "B", "C"])

    contributions = explainer.produce(x_one_dim)[0]

    assert contributions.shape == (3, 3)
    assert contributions["A"][0] == 0
    assert contributions["B"][0] == 2
    assert contributions["C"][0] == 0


def test_produce_occlusion_classification_no_transforms_remove(classification_no_transforms):
    model = classification_no_transforms
    explainer = UnivariateOcclusionSaliency(
        model=model["model"],
        x_train_orig=model["x"],
        width=1,
        k="remove",
        transformers=model["transformers"],
        fit_on_init=True,
        classes=np.arange(1, 4),
    )

    x_one_dim = pd.DataFrame([[1, 1, 1]], columns=["A", "B", "C"])

    contributions = explainer.produce(x_one_dim)[0]

    assert contributions.shape == (3, 3)
    assert contributions["A"][0] == 0
    assert abs(contributions["B"][0]) < 0.01
    assert contributions["C"][0] == 0
