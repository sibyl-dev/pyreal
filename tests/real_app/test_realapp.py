import pytest
from pyreal import RealApp
import numpy as np


def test_initalization_one_model(regression_one_hot):
    realApp = RealApp(
        regression_one_hot["model"], regression_one_hot["x"],
        transformers=regression_one_hot["transformers"]
    )
    assert realApp.get_active_model() is regression_one_hot["model"]


def test_initalization_model_list(regression_one_hot, regression_no_transforms):
    realApp = RealApp(
        [regression_one_hot["model"], regression_no_transforms["model"]], regression_one_hot["x"],
        transformers=regression_one_hot["transformers"]
    )
    assert realApp.get_active_model() is regression_one_hot["model"]


def test_initalization_model_dict(regression_one_hot, regression_no_transforms):
    model_dict = {"id1": regression_one_hot["model"], "id2": regression_no_transforms["model"]}
    realApp = RealApp(
        model_dict, regression_one_hot["x"],
        transformers=regression_one_hot["transformers"]
    )
    assert realApp.get_active_model() is regression_one_hot["model"]

    realApp = RealApp(
        model_dict, regression_one_hot["x"],
        transformers=regression_one_hot["transformers"],
        active_model_id="id2"
    )
    assert realApp.get_active_model() is regression_no_transforms["model"]


def test_add_model(regression_one_hot, regression_no_transforms):
    realApp = RealApp(
        regression_one_hot["model"], regression_one_hot["x"],
        transformers=regression_one_hot["transformers"]
    )

    realApp.add_model(regression_no_transforms["model"])
    assert len(realApp.models) == 2


def test_set_active_model(regression_one_hot, regression_no_transforms):
    realApp = RealApp(
        regression_one_hot["model"], regression_one_hot["x"],
        transformers=regression_one_hot["transformers"]
    )

    realApp.add_model(regression_no_transforms["model"], "id2")
    realApp.set_active_model_id("id2")

    assert len(realApp.models) == 2
    assert realApp.get_active_model() is regression_no_transforms["model"]


def test_predict(regression_one_hot, regression_no_transforms):
    realApp = RealApp(
        regression_one_hot["model"], regression_one_hot["x"],
        transformers=regression_one_hot["transformers"]
    )
    print(realApp.explainers)

    expected = np.array(regression_one_hot["y"]).reshape(-1)
    result = realApp.predict(regression_one_hot["x"])
    assert np.array_equal(result, expected)
