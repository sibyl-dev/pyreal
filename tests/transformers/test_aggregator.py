from pyreal.transformers.aggregator import Mappings, Aggregator
import pandas as pd
import pytest
from pyreal.explanation_types.feature_based import (
    AdditiveFeatureImportanceExplanation,
    AdditiveFeatureContributionExplanation,
)


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


def test_fit(mapping_input):
    agg = Aggregator(mapping_input)
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
