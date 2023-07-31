import pickle

import numpy as np
import pandas as pd
import pytest

from pyreal import RealApp


def test_initalization_one_model(regression_one_hot):
    real_app = RealApp(
        regression_one_hot["model"],
        regression_one_hot["x"],
        transformers=regression_one_hot["transformers"],
    )
    assert real_app.get_active_model() is regression_one_hot["model"]


def test_initalization_model_list(regression_one_hot, regression_no_transforms):
    real_app = RealApp(
        [regression_one_hot["model"], regression_no_transforms["model"]],
        regression_one_hot["x"],
        transformers=regression_one_hot["transformers"],
    )
    assert real_app.get_active_model() is regression_one_hot["model"]


def test_initalization_model_dict(regression_one_hot, regression_no_transforms):
    model_dict = {"id1": regression_one_hot["model"], "id2": regression_no_transforms["model"]}
    real_app = RealApp(
        model_dict, regression_one_hot["x"], transformers=regression_one_hot["transformers"]
    )
    assert real_app.get_active_model() is regression_one_hot["model"]

    real_app = RealApp(
        model_dict,
        regression_one_hot["x"],
        transformers=regression_one_hot["transformers"],
        active_model_id="id2",
    )
    assert real_app.get_active_model() is regression_no_transforms["model"]


def test_add_model(regression_one_hot, regression_no_transforms):
    real_app = RealApp(
        regression_one_hot["model"],
        regression_one_hot["x"],
        transformers=regression_one_hot["transformers"],
    )

    real_app.add_model(regression_no_transforms["model"])
    assert len(real_app.models) == 2


def test_set_active_model(regression_one_hot, regression_no_transforms):
    real_app = RealApp(
        regression_one_hot["model"],
        regression_one_hot["x"],
        transformers=regression_one_hot["transformers"],
    )

    real_app.add_model(regression_no_transforms["model"], "id2")
    real_app.set_active_model_id("id2")

    assert len(real_app.models) == 2
    assert real_app.get_active_model() is regression_no_transforms["model"]


def test_predict(regression_one_hot):
    real_app = RealApp(
        regression_one_hot["model"],
        regression_one_hot["x"],
        transformers=regression_one_hot["transformers"],
    )

    expected = {
        key: value
        for (key, value) in zip(
            regression_one_hot["x"].index, np.array(regression_one_hot["y"]).reshape(-1)
        )
    }
    result = real_app.predict(regression_one_hot["x"])
    assert np.array_equal(result, expected)


def test_predict_multiple_models(dummy_models):
    x = pd.DataFrame([[1, 0]])
    real_app = RealApp(dummy_models, x, active_model_id="id2")

    expected = {0: 3}
    result = real_app.predict(x)
    assert np.array_equal(result, expected)

    expected = {0: 2}
    result = real_app.predict(x, model_id="id1")
    assert np.array_equal(result, expected)


def test_realapp_check_size(regression_no_transforms):
    x_large = pd.DataFrame(np.random.randint(0, 100, (1000, 3)))
    y_large = pd.DataFrame(np.random.randint(0, 100, (1000,)))
    realapp = RealApp(regression_no_transforms["model"])
    realapp_size = len(pickle.dumps(realapp))
    assert realapp_size < 2000

    realapp.prepare_feature_importance(
        x_train_orig=x_large, y_train=y_large, algorithm="permutation"
    )
    assert realapp_size < 2000


def test_no_dataset_on_init_or_fit_ensure_break(regression_no_transforms):
    model = regression_no_transforms["model"]
    x = regression_no_transforms["x"]
    explainer = RealApp(model)
    with pytest.raises(ValueError):
        explainer.produce_feature_contributions(x)
    with pytest.raises(ValueError):
        explainer.prepare_feature_importance()
