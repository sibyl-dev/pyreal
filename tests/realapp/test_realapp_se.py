import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from pyreal import RealApp


def test_produce_similar_examples(regression_one_hot_with_interpret):
    x = pd.DataFrame([[2, 0, 0], [6, 6, 6], [6, 7, 8], [2, 0, 1]], columns=["A", "B", "C"])
    y = pd.Series([1, 0, 0, 1])
    real_app = RealApp(
        regression_one_hot_with_interpret["model"],
        X_train_orig=x,
        y_train=y,
        transformers=regression_one_hot_with_interpret["transformers"],
        id_column="ID",
    )

    input_rows = pd.DataFrame([["id1", 6, 7, 9], ["id2", 2, 1, 1]], columns=["ID", "A", "B", "C"])
    explanation = real_app.produce_similar_examples(input_rows, n=2)
    assert "id1" in explanation
    assert_frame_equal(explanation["id1"]["X"], x.iloc[[2, 1], :] + 1)
    assert_series_equal(explanation["id1"]["y"], y.iloc[[2, 1]])
    assert_series_equal(explanation["id1"]["Input"], input_rows.iloc[0, 1:] + 1, check_dtype=False)

    assert "id2" in explanation
    assert_frame_equal(explanation["id2"]["X"], x.iloc[[3, 0], :] + 1)
    assert_series_equal(explanation["id2"]["y"], y.iloc[[3, 0]])


def test_prepare_similar_examples(regression_no_transforms):
    real_app = RealApp(
        regression_no_transforms["model"],
        transformers=regression_no_transforms["transformers"],
    )
    x = pd.DataFrame([[2, 10, 10]])

    with pytest.raises(ValueError):
        real_app.produce_similar_examples(x)

    # Confirm no error
    real_app.prepare_similar_examples(
        x_train_orig=regression_no_transforms["x"], y_train=regression_no_transforms["y"]
    )

    # Confirm explainer was prepped and now works without being given data
    real_app.produce_similar_examples(x)


def test_prepare_similar_examples_with_id_column(regression_no_transforms):
    real_app = RealApp(
        regression_no_transforms["model"],
        transformers=regression_no_transforms["transformers"],
        id_column="ID",
    )
    features = ["A", "B", "C"]
    x_multi_dim = pd.DataFrame(
        [[4, 1, 1, "a"], [6, 2, 3, "b"], [1, 1, 1, "c"]], columns=features + ["ID"]
    )

    # Confirm no error
    real_app.prepare_similar_examples(x_train_orig=x_multi_dim, y_train=pd.Series([0, 1, 1]))

    # Confirm explainer was prepped and now works without being given data
    real_app.produce_similar_examples(x_multi_dim, n=1)


def test_produce_similar_examples_with_standardization(dummy_model):
    x = pd.DataFrame([[1, 100], [3, 100], [1, 200], [3, 300]])
    y = pd.Series([1, 2, 3, 4])
    real_app = RealApp(
        dummy_model,
        X_train_orig=x,
        y_train=y,
    )

    explanation = real_app.produce_similar_examples(pd.Series([1, 100]), n=2, standardize=True)

    assert_frame_equal(explanation["X"], x.iloc[[0, 2], :])
    assert_series_equal(explanation["y"], y.iloc[[0, 2]])


def test_produce_similar_examples_with_format_func(regression_one_hot_with_interpret):
    def format_func(pred):
        return "test%i" % pred

    x = pd.DataFrame([[2, 0, 0], [6, 6, 6], [6, 7, 8], [2, 0, 1]], columns=["A", "B", "C"])
    y = pd.Series([1, 0, 0, 1])
    real_app = RealApp(
        regression_one_hot_with_interpret["model"],
        X_train_orig=x,
        y_train=y,
        transformers=regression_one_hot_with_interpret["transformers"],
        id_column="ID",
        pred_format_func=format_func,
    )

    input_rows = pd.DataFrame([["id1", 6, 7, 9], ["id2", 2, 1, 1]], columns=["ID", "A", "B", "C"])
    explanation = real_app.produce_similar_examples(input_rows, n=2)
    assert_series_equal(explanation["id1"]["y"], y.iloc[[2, 1]].apply(format_func))
    explanation = real_app.produce_similar_examples(input_rows, n=2, format_y=False)
    assert_series_equal(explanation["id1"]["y"], y.iloc[[2, 1]])
