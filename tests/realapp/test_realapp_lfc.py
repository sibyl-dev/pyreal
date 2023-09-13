import numpy as np
import pandas as pd
import pytest

from pyreal import RealApp
from pyreal.realapp.realapp import _get_average_or_mode


def test_average_or_mode():
    mix = pd.DataFrame([[1, "a", 2, "a"], [1, "a", 4, "a"], [1, "b", 3, "a"]])
    expected = [1, "a", 3, "a"]
    result = _get_average_or_mode(mix)
    for i in range(len(expected)):
        assert expected[i] == result[i]

    mode_only = pd.DataFrame([["a", "a", "b", "d"], ["b", "a", "b", "d"], ["a", "b", "d", "d"]])
    expected = ["a", "a", "b", "d"]
    result = _get_average_or_mode(mode_only)
    for i in range(len(expected)):
        assert expected[i] == result[i]

    mean_only = pd.DataFrame([[1, 0, 2, 2], [1, 5, 3, 3], [1, -5, 4, 4]])
    expected = [1, 0, 3, 3]
    result = _get_average_or_mode(mean_only)
    for i in range(len(expected)):
        assert expected[i] == result[i]


def test_prepare_local_feature_contribution(regression_no_transforms):
    real_app = RealApp(
        regression_no_transforms["model"],
        transformers=regression_no_transforms["transformers"],
    )
    x = pd.DataFrame([[2, 10, 10]])

    with pytest.raises(ValueError):
        real_app.produce_feature_contributions(x)

    # Confirm no error
    real_app.prepare_feature_contributions(
        x_train_orig=regression_no_transforms["x"], y_train=regression_no_transforms["y"]
    )

    # Confirm explainer was prepped and now works without being given data
    real_app.produce_feature_contributions(x)


def test_prepare_local_feature_contribution_with_id_column(regression_no_transforms):
    real_app = RealApp(
        regression_no_transforms["model"],
        transformers=regression_no_transforms["transformers"],
        id_column="ID",
    )
    features = ["A", "B", "C"]
    x_multi_dim = pd.DataFrame([[4, 1, 1, "a"], [6, 2, 3, "b"]], columns=features + ["ID"])

    # Confirm no error
    real_app.prepare_feature_contributions(
        x_train_orig=x_multi_dim, y_train=regression_no_transforms["y"]
    )

    # Confirm explainer was prepped and now works without being given data
    real_app.produce_feature_contributions(x_multi_dim)


def test_produce_local_feature_contributions(regression_no_transforms):
    real_app = RealApp(
        regression_no_transforms["model"],
        regression_no_transforms["x"],
        transformers=regression_no_transforms["transformers"],
    )
    features = ["A", "B", "C"]

    # x_one_dim = pd.DataFrame([[2, 10, 10]], columns=features)
    x_one_dim = pd.Series([2, 10, 10], index=features)

    expected = np.mean(regression_no_transforms["y"])
    explanation = real_app.produce_feature_contributions(x_one_dim)

    assert list(explanation["Feature Name"]) == features
    assert list(explanation["Feature Value"]) == list(x_one_dim)
    assert list(explanation["Contribution"]) == [x_one_dim.iloc[0] - expected, 0, 0]
    assert list(explanation["Average/Mode"]) == list(x_one_dim)

    x_multi_dim = pd.DataFrame([[2, 1, 1], [4, 2, 3]], columns=features)
    explanation = real_app.produce_feature_contributions(x_multi_dim)

    assert list(explanation[0]["Feature Name"]) == features
    assert list(explanation[0]["Feature Value"]) == list(x_multi_dim.iloc[0])
    assert list(explanation[0]["Contribution"]) == [x_multi_dim.iloc[0, 0] - expected, 0, 0]
    assert list(explanation[0]["Average/Mode"]) == [3, 1.5, 2]

    assert list(explanation[1]["Feature Name"]) == features
    assert list(explanation[1]["Feature Value"]) == list(x_multi_dim.iloc[1])
    assert list(explanation[1]["Contribution"]) == [x_multi_dim.iloc[1, 0] - expected, 0, 0]
    assert list(explanation[1]["Average/Mode"]) == [3, 1.5, 2]


def test_produce_local_feature_contributions_with_id_column(regression_one_hot):
    real_app = RealApp(
        regression_one_hot["model"],
        regression_one_hot["x"],
        transformers=regression_one_hot["transformers"],
        id_column="ID",
    )

    features = ["A", "B", "C"]
    x_one_dim = pd.Series([4, 1, 1, "ab"], index=features + ["ID"])
    explanation = real_app.produce_feature_contributions(x_one_dim)
    explanation_a1 = explanation.sort_values(by="Feature Name", axis=0)

    x_multi_dim = pd.DataFrame([[4, 1, 1, "a"], [6, 2, 3, "b"]], columns=features + ["ID"])

    explanation = real_app.produce_feature_contributions(x_multi_dim)

    explanation_a2 = explanation["a"].sort_values(by="Feature Name", axis=0)
    explanation_b = explanation["b"].sort_values(by="Feature Name", axis=0)

    for explanation_a in [explanation_a1, explanation_a2]:
        assert list(explanation_a["Feature Name"]) == features
        assert list(explanation_a["Feature Value"]) == list(x_multi_dim.iloc[0])[:-1]
        for num in list(explanation_a["Contribution"]):
            assert abs(num) < 0.001
    assert list(explanation_a1["Average/Mode"]) == [4, 1, 1]
    assert list(explanation_a2["Average/Mode"]) == [5, 1.5, 2]

    assert list(explanation_b["Feature Name"]) == features
    assert list(explanation_b["Feature Value"]) == list(x_multi_dim.iloc[1])[:-1]
    assert abs(list(explanation_b["Contribution"])[0] - 1) < 0.001
    for num in list(explanation_a["Contribution"][1:]):
        assert abs(num) < 0.001
    assert list(explanation_b["Average/Mode"]) == [5, 1.5, 2]


def test_produce_local_feature_contributions_no_data_on_init(regression_no_transforms):
    real_app = RealApp(
        regression_no_transforms["model"],
        transformers=regression_no_transforms["transformers"],
    )
    features = ["A", "B", "C"]
    x_one_dim = pd.DataFrame([[2, 10, 10]], columns=features)

    expected = np.mean(regression_no_transforms["y"])
    explanation = real_app.produce_feature_contributions(
        x_one_dim, x_train_orig=regression_no_transforms["x"]
    )

    assert list(explanation[0]["Feature Name"]) == features
    assert list(explanation[0]["Feature Value"]) == list(x_one_dim.iloc[0])
    assert list(explanation[0]["Contribution"]) == [x_one_dim.iloc[0, 0] - expected, 0, 0]
    assert list(explanation[0]["Average/Mode"]) == list(x_one_dim.iloc[0])


def test_produce_local_feature_contributions_num(regression_no_transforms):
    real_app = RealApp(
        regression_no_transforms["model"],
        regression_no_transforms["x"],
        transformers=regression_no_transforms["transformers"],
    )

    x_one_dim = pd.DataFrame([[2, 10, 10]])
    explanation = real_app.produce_feature_contributions(
        x_one_dim, num_features=2, select_by="min"
    )
    assert explanation[0].shape == (2, 4)
