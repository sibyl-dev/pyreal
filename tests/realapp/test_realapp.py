import pickle

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_index_equal

from pyreal import RealApp
from pyreal.transformers import TransformerBase


def test_initialization_one_model(regression_one_hot):
    real_app = RealApp(
        regression_one_hot["model"],
        regression_one_hot["x"],
        transformers=regression_one_hot["transformers"],
    )
    assert real_app.get_active_model() is regression_one_hot["model"]


def test_initialization_model_list(regression_one_hot, regression_no_transforms):
    real_app = RealApp(
        [regression_one_hot["model"], regression_no_transforms["model"]],
        regression_one_hot["x"],
        transformers=regression_one_hot["transformers"],
    )
    assert real_app.get_active_model() is regression_one_hot["model"]


def test_initialization_model_dict(regression_one_hot, regression_no_transforms):
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

    result = real_app.predict(regression_one_hot["x"], as_dict=False)
    expected = np.array(regression_one_hot["y"]).reshape(-1)
    assert np.array_equal(result, expected)


def test_predict_proba(classification_no_transforms):
    real_app = RealApp(
        classification_no_transforms["model"],
        classification_no_transforms["x"],
        transformers=classification_no_transforms["transformers"],
        classes=classification_no_transforms["classes"],
    )

    quantity = (
        classification_no_transforms["coefs"].T @ classification_no_transforms["x"].to_numpy()
    )
    expected_probs = np.exp(quantity) / np.sum(np.exp(quantity), axis=1, keepdims=True)

    result = real_app.predict_proba(classification_no_transforms["x"])
    for key in result.keys():
        assert np.allclose(result[key], expected_probs[key, :])

    result = real_app.predict_proba(classification_no_transforms["x"], as_dict=False)
    assert np.allclose(result, expected_probs)


def test_predict_series(regression_one_hot):
    real_app = RealApp(
        regression_one_hot["model"],
        regression_one_hot["x"],
        transformers=regression_one_hot["transformers"],
    )

    expected = np.array(regression_one_hot["y"])[0]
    result = real_app.predict(regression_one_hot["x"].iloc[0])
    assert np.array_equal(result, expected)


def test_predict_id_column(dummy_model):
    x = pd.DataFrame([[1, 0], [2, 2]])
    real_app = RealApp(
        dummy_model,
        x,
        id_column="ID",
    )
    features = ["A", "B"]
    x_multi_dim = pd.DataFrame([[4, 1, "a"], [6, 2, "b"]], columns=features + ["ID"])

    expected = {"a": 5, "b": 8}
    result = real_app.predict(x_multi_dim)
    assert np.array_equal(result, expected)

    expected_single = 5
    result_single = real_app.predict(x_multi_dim.iloc[0])
    assert expected_single == result_single[0]


def test_predict_multiple_models(dummy_models):
    x = pd.DataFrame([[1, 0]])
    real_app = RealApp(dummy_models, x, active_model_id="id2")

    expected = {0: 3}
    result = real_app.predict(x)
    assert np.array_equal(result, expected)

    expected = {0: 2}
    result = real_app.predict(x, model_id="id1")
    assert np.array_equal(result, expected)


def test_predict_format(regression_one_hot):
    def format_func(pred):
        return "test%i" % pred

    real_app = RealApp(
        regression_one_hot["model"],
        regression_one_hot["x"],
        transformers=regression_one_hot["transformers"],
        pred_format_func=format_func,
    )

    expected = {
        key: format_func(value)
        for (key, value) in zip(
            regression_one_hot["x"].index,
            np.array(regression_one_hot["y"]).reshape(-1),
        )
    }
    result = real_app.predict(regression_one_hot["x"])
    assert np.array_equal(result, expected)

    # Test with format=false
    expected_unformatted = {
        key: value
        for (key, value) in zip(
            regression_one_hot["x"].index,
            np.array(regression_one_hot["y"]).reshape(-1),
        )
    }
    result = real_app.predict(regression_one_hot["x"], format=False)
    assert np.array_equal(result, expected_unformatted)

    # Test with as_dict=false
    result = real_app.predict(regression_one_hot["x"], as_dict=False)
    expected = np.array(list(expected.values()))
    assert np.array_equal(result, expected)

    # Test series input
    expected = np.array([expected[0]])
    result = real_app.predict(regression_one_hot["x"].iloc[0])
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


def test_fit_transformers(dummy_model):
    class DummyTransformer(TransformerBase):
        def __init__(self, **kwargs):
            self.columns = None
            super().__init__(**kwargs)

        def fit(self, x):
            self.columns = x.columns

        def data_transform(self, x):
            return x

    x = pd.DataFrame([[1, 2, "ab"], [1, 2, ["bc"]]], columns=["A", "B", "ID"])
    transformer = DummyTransformer()
    RealApp(dummy_model, x, transformers=transformer, fit_transformers=True)
    assert_index_equal(transformer.columns, x.columns)

    transformer = DummyTransformer()
    RealApp(dummy_model, x, transformers=transformer, fit_transformers=True, id_column="ID")
    assert_index_equal(transformer.columns, x.drop(columns="ID").columns)
