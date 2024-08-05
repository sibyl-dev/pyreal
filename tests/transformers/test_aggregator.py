import numpy as np
import pandas as pd
import pytest

from pyreal.explainers import LocalFeatureContribution
from pyreal.explanation_types.feature_based import (
    AdditiveFeatureContributionExplanation,
    AdditiveFeatureImportanceExplanation,
)
from pyreal.transformers import FeatureSelectTransformer
from pyreal.transformers.aggregator import Aggregator, Mappings


@pytest.fixture
def mappings():
    one_to_many = {
        "parent1": ["child1", "child2"],
        "parent2": ["child3"],
        "parent3": ["child4", "child5", "child6"],
    }

    return Mappings.generate_mappings(one_to_many=one_to_many)


def test_generate_mappings():
    one_to_many = {
        "parent1": ["child1", "child2"],
        "parent2": ["child3"],
        "parent3": ["child4", "child5", "child6"],
    }
    many_to_one = {
        "child1": "parent1",
        "child2": "parent1",
        "child3": "parent2",
        "child4": "parent3",
        "child5": "parent3",
        "child6": "parent3",
    }

    dataframe = pd.DataFrame(
        [
            ["parent1", "child1"],
            ["parent1", "child2"],
            ["parent2", "child3"],
            ["parent3", "child4"],
            ["parent3", "child5"],
            ["parent3", "child6"],
        ],
        columns=["parent", "child"],
    )

    test_mappings = Mappings.generate_mappings(one_to_many=one_to_many)
    assert test_mappings.one_to_many == one_to_many
    assert test_mappings.many_to_one == many_to_one

    test_mappings = Mappings.generate_mappings(many_to_one=many_to_one)
    assert test_mappings.one_to_many == one_to_many
    assert test_mappings.many_to_one == many_to_one

    test_mappings = Mappings.generate_mappings(dataframe=dataframe)
    assert test_mappings.one_to_many == one_to_many
    assert test_mappings.many_to_one == many_to_one


def test_fit(mappings):
    agg = Aggregator(mappings)
    assert agg.fit(pd.DataFrame([])) == agg  # aggregator does not define custom fit


def test_transform_basic_string_inputs():
    small_mappings = Mappings.generate_mappings(one_to_many={"parent1": ["child1", "child2"]})
    x = pd.DataFrame([[1, 2], [3, 4]], columns=["child1", "child2"])

    agg = Aggregator(small_mappings, func="sum")
    result = agg.transform(x)
    expected = pd.DataFrame([[3], [7]], columns=["parent1"])
    pd.testing.assert_frame_equal(result, expected)

    agg = Aggregator(small_mappings, func="mean")
    result = agg.transform(x)
    expected = pd.DataFrame([[1.5], [3.5]], columns=["parent1"])
    pd.testing.assert_frame_equal(result, expected)

    agg = Aggregator(small_mappings, func="max")
    result = agg.transform(x)
    expected = pd.DataFrame([[2], [4]], columns=["parent1"])
    pd.testing.assert_frame_equal(result, expected)

    agg = Aggregator(small_mappings, func="min")
    result = agg.transform(x)
    expected = pd.DataFrame([[1], [3]], columns=["parent1"])
    pd.testing.assert_frame_equal(result, expected)


def test_transform_remove(mappings):
    x = pd.DataFrame(
        [[1, 2, 3, 4, 5, 6, 0], [7, 8, 9, 10, 11, 12, 0]],
        columns=[f"child{i}" for i in range(1, 7)] + ["extra"],
    )

    agg = Aggregator(mappings, func="remove")
    result = agg.transform(x)
    expected = pd.DataFrame(
        [[None, None, None, 0], [None, None, None, 0]],
        columns=["parent1", "parent2", "parent3", "extra"],
    )
    pd.testing.assert_frame_equal(result[["parent1", "parent2", "parent3", "extra"]], expected)


def test_transform_extra_columns(mappings):
    small_mappings = Mappings.generate_mappings(one_to_many={"parent1": ["child1", "child2"]})
    x = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=["child1", "child2", "extra"])

    agg = Aggregator(small_mappings, func=sum)
    result = agg.transform(x)
    expected = pd.DataFrame([[3, 3], [9, 6]], columns=["parent1", "extra"])
    pd.testing.assert_frame_equal(result[["parent1", "extra"]], expected)

    agg = Aggregator(small_mappings, func=max)
    result = agg.transform(x)
    expected = pd.DataFrame([[2, 3], [5, 6]], columns=["parent1", "extra"])
    pd.testing.assert_frame_equal(result[["parent1", "extra"]], expected)

    agg = Aggregator(small_mappings, func=min)
    result = agg.transform(x)
    expected = pd.DataFrame([[1, 3], [4, 6]], columns=["parent1", "extra"])
    pd.testing.assert_frame_equal(result[["parent1", "extra"]], expected)


def test_transform_missing_parent(mappings):
    x = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=["child1", "child2", "child3"])

    agg = Aggregator(mappings, func=sum, missing="ignore")
    result = agg.transform(x)
    expected = pd.DataFrame([[3, 3], [9, 6]], columns=["parent1", "parent2"])
    pd.testing.assert_frame_equal(result[["parent1", "parent2"]], expected)

    agg = Aggregator(mappings, func=sum, missing="raise")
    with pytest.raises(ValueError):
        agg.transform(x)


def test_transform_missing_child(mappings):
    x = pd.DataFrame(
        [[1, 2, 3, 4], [5, 6, 7, 8]], columns=["child1", "child2", "child4", "child5"]
    )

    agg = Aggregator(mappings, func=sum, missing="ignore")
    result = agg.transform(x)
    expected = pd.DataFrame([[3, 7], [11, 15]], columns=["parent1", "parent3"])
    pd.testing.assert_frame_equal(result[["parent1", "parent3"]], expected)

    agg = Aggregator(mappings, func=sum, missing="raise")
    with pytest.raises(ValueError):
        agg.transform(x)


def test_transform_no_drop(mappings):
    x = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=["child1", "child2", "extra"])

    agg = Aggregator(mappings, func=sum, drop_original=False)
    result = agg.transform(x)
    expected = pd.DataFrame(
        [[1, 2, 3, 3], [4, 5, 6, 9]], columns=["child1", "child2", "extra", "parent1"]
    )
    print(result)
    pd.testing.assert_frame_equal(result[["child1", "child2", "extra", "parent1"]], expected)


def test_transform_additive_contributions(mappings):
    contributions = pd.DataFrame(
        [[1, 2, 3, 10], [4, 5, 6, 10]], columns=["child1", "child2", "child3", "extra"]
    )
    explanation = AdditiveFeatureContributionExplanation(contributions)

    agg = Aggregator(mappings, func=max)
    result = agg.transform_explanation(explanation).get()

    expected = pd.DataFrame([[3, 3, 10], [9, 6, 10]], columns=["parent1", "parent2", "extra"])

    pd.testing.assert_frame_equal(result[["parent1", "parent2", "extra"]], expected)


def test_transform_additive_importance(mappings):
    importance = pd.DataFrame([[1, 2, 3, 10]], columns=["child1", "child2", "child3", "extra"])
    explanation = AdditiveFeatureImportanceExplanation(importance)
    agg = Aggregator(mappings, func=max)
    result = agg.transform_explanation(explanation).get()

    expected = pd.DataFrame([[3, 3, 10]], columns=["parent1", "parent2", "extra"])
    pd.testing.assert_frame_equal(result[["parent1", "parent2", "extra"]], expected)


def test_transform_additive_contributions_raise(mappings):
    contributions = pd.DataFrame(
        [[1, 2, 3, 10], [4, 5, 6, 10]], columns=["child1", "child2", "child3", "extra"]
    )
    explanation = AdditiveFeatureContributionExplanation(contributions)

    agg = Aggregator(mappings, func=max, missing="raise")
    with pytest.raises(ValueError):
        agg.transform_explanation(explanation).get()


def test_in_explainer(regression_no_transforms):
    small_mappings = Mappings.generate_mappings(one_to_many={"A": ["a", "aa"]})
    x = pd.DataFrame([[2, 2, 1, 3], [4, 4, 3, 4], [6, 6, 7, 2]], columns=["a", "aa", "B", "C"])
    exp = LocalFeatureContribution(
        model=regression_no_transforms["model"],
        x_train_orig=x,
        y_train=regression_no_transforms["y"],
        transformers=[
            Aggregator(small_mappings, func="sum"),
            FeatureSelectTransformer(columns=["A", "B", "C"]),
        ],
        fit_on_init=True,
        fit_transformers=True,
        e_algorithm="shap",
    )
    result = exp.produce_explanation_interpret(x)

    expected_values = pd.DataFrame([[4, 1, 3], [8, 3, 4], [12, 7, 2]], columns=["A", "B", "C"])
    pd.testing.assert_frame_equal(result.get_values()[["A", "B", "C"]], expected_values)

    assert (result.get()[["B", "C"]] == 0).all().all()
    expected = 8.0  # Mean of resulting A values in training set
    pd.testing.assert_series_equal(result.get()["A"], pd.Series([4, 8, 12], name="A") - expected)


def test_in_explainer_interpret(regression_no_transforms):
    small_mappings = Mappings.generate_mappings(
        one_to_many={"AB": ["A", "B"], "DE": ["D", "E", "F"]}
    )
    exp = LocalFeatureContribution(
        model=regression_no_transforms["model"],
        x_train_orig=regression_no_transforms["x"],
        y_train=regression_no_transforms["y"],
        transformers=[
            Aggregator(small_mappings, func="remove", interpret=True, model=False),
        ],
        fit_on_init=True,
        fit_transformers=True,
        e_algorithm="shap",
    )
    result = exp.produce_explanation_interpret(regression_no_transforms["x"])

    expected_values = pd.DataFrame([[None, 3], [None, 4], [None, 2]], columns=["AB", "C"])
    pd.testing.assert_frame_equal(result.get_values()[["AB", "C"]], expected_values)

    assert (result.get()["C"] == 0).all()

    expected_contributions = regression_no_transforms["x"]["A"] - np.mean(
        regression_no_transforms["y"]
    )
    expected_contributions.name = "AB"
    pd.testing.assert_series_equal(result.get()["AB"], expected_contributions)
