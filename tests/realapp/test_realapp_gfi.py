import pandas as pd
import pytest
from pandas.testing import assert_index_equal

from pyreal import RealApp


def test_prepare_global_feature_importance(regression_no_transforms):
    realApp = RealApp(
        regression_no_transforms["model"],
        transformers=regression_no_transforms["transformers"],
    )

    with pytest.raises(ValueError):
        realApp.produce_feature_importance()

    # Confirm no error
    realApp.prepare_feature_importance(
        x_train_orig=regression_no_transforms["x"], y_train=regression_no_transforms["y"]
    )

    # Confirm explainer was prepped and now works without being given data
    realApp.produce_feature_importance()


def test_prepare_global_feature_importance_with_id_column(regression_no_transforms):
    realApp = RealApp(
        regression_no_transforms["model"],
        transformers=regression_no_transforms["transformers"],
        id_column="ID",
    )
    features = ["A", "B", "C"]
    x_multi_dim = pd.DataFrame([[4, 1, 1, "a"], [6, 2, 3, "b"]], columns=features + ["ID"])

    # Confirm no error
    realApp.prepare_feature_importance(
        x_train_orig=x_multi_dim, y_train=regression_no_transforms["y"]
    )

    # Confirm explainer was prepped and now works without being given data
    realApp.produce_feature_importance()


def test_produce_global_feature_importance(regression_no_transforms):
    realApp = RealApp(
        regression_no_transforms["model"],
        regression_no_transforms["x"],
        y_train=regression_no_transforms["y"],
        transformers=regression_no_transforms["transformers"],
    )
    features = ["A", "B", "C"]

    explanation = realApp.produce_feature_importance()

    assert list(explanation["Feature Name"]) == features
    assert list(explanation["Importance"]) == [4 / 3, 0, 0]

    explanation = realApp.produce_feature_importance(algorithm="permutation")
    assert list(explanation["Feature Name"]) == features
    assert abs(list(explanation["Importance"])[0]) > 0.1
    assert list(explanation["Importance"])[1:] == [0, 0]

    # confirm no bug in explainer caching
    realApp.produce_feature_importance(algorithm="permutation")


def test_produce_global_feature_importance_no_format(regression_no_transforms):
    realApp = RealApp(
        regression_no_transforms["model"],
        regression_no_transforms["x"],
        y_train=regression_no_transforms["y"],
        transformers=regression_no_transforms["transformers"],
    )

    explanation = realApp.produce_feature_importance(format_output=False)
    assert explanation.shape == (1, regression_no_transforms["x"].shape[1])
    assert_index_equal(explanation.columns, regression_no_transforms["x"].columns)
    assert list(explanation.iloc[0]) == [4 / 3, 0, 0]


def test_produce_global_feature_importance_no_data_on_init(regression_no_transforms):
    realApp = RealApp(
        regression_no_transforms["model"],
        transformers=regression_no_transforms["transformers"],
    )
    features = ["A", "B", "C"]

    explanation = realApp.produce_feature_importance(
        x_train_orig=regression_no_transforms["x"], y_train=regression_no_transforms["y"]
    )

    assert list(explanation["Feature Name"]) == features
    assert list(explanation["Importance"]) == [4 / 3, 0, 0]

    explanation = realApp.produce_feature_importance(
        algorithm="permutation",
        x_train_orig=regression_no_transforms["x"],
        y_train=regression_no_transforms["y"],
    )
    assert list(explanation["Feature Name"]) == features
    assert abs(list(explanation["Importance"])[0]) > 0.1
    assert list(explanation["Importance"])[1:] == [0, 0]

    # confirm no bug in explainer caching
    realApp.produce_feature_importance(algorithm="permutation")


def test_produce_global_importance_contributions_num(regression_no_transforms):
    realApp = RealApp(
        regression_no_transforms["model"],
        regression_no_transforms["x"],
        transformers=regression_no_transforms["transformers"],
    )

    explanation = realApp.produce_feature_importance(num_features=2, select_by="absolute")
    assert explanation.shape == (2, 2)
