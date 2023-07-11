import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

from pyreal.explainers.example.similar_examples import SimilarExamples


def test_produce(dummy_model):
    x = pd.DataFrame([[1, 1, 1], [4, 5, 3], [0, 0, 0], [5, 5, 3]])
    y = pd.Series([0, 1, 0, 1])

    explainer = SimilarExamples(model=dummy_model, x_train_orig=x, y_train=y, fit_on_init=True)
    result = explainer.produce(pd.DataFrame([[0, 1, 0]]), n=2)
    expected_examples = x.iloc[[2, 0], :]
    expected_targets = y.iloc[[2, 0]]
    assert len(result.get_row_ids()) == 1
    assert result.get_examples(row_id=0).shape[0] == 2
    assert_frame_equal(result.get_examples(), expected_examples)
    assert_series_equal(result.get_targets(), expected_targets)


def test_produce_with_transforms(regression_one_hot_with_interpret):
    x = pd.DataFrame([[2, 1, 3], [4, 3, 4], [6, 7, 10]], columns=["A", "B", "C"])
    y = pd.Series([1, 2, 3])
    explainer = SimilarExamples(
        model=regression_one_hot_with_interpret["model"],
        x_train_orig=x,
        y_train=y,
        transformers=regression_one_hot_with_interpret["transformers"],
        fit_on_init=True,
        feature_descriptions={"A": "Feature A"},
    )
    result = explainer.produce(pd.DataFrame([[2, 1, 4]], columns=["A", "B", "C"]), n=1)
    expected_examples = pd.DataFrame([((x.iloc[0, :]) + 1).rename({"A": "Feature A"})])
    expected_targets = pd.Series([y.iloc[0]])

    assert len(result.get_row_ids()) == 1
    assert result.get_examples(row_id=0).shape[0] == 1
    assert_frame_equal(result.get_examples(), expected_examples)
    assert_series_equal(result.get_targets(), expected_targets)


def test_produce_multiple_with_transforms(regression_one_hot_with_interpret):
    x = pd.DataFrame([[2, 1, 3], [4, 3, 4], [6, 7, 10]], columns=["A", "B", "C"])
    y = pd.Series([1, 2, 3])
    explainer = SimilarExamples(
        model=regression_one_hot_with_interpret["model"],
        x_train_orig=x,
        y_train=y,
        transformers=regression_one_hot_with_interpret["transformers"],
        fit_on_init=True,
        feature_descriptions={"A": "Feature A"},
    )
    result = explainer.produce(pd.DataFrame([[2, 1, 4], [6, 7, 9]], columns=["A", "B", "C"]), n=2)
    expected_examples_1 = (x.iloc[[0, 1], :] + 1).rename(columns={"A": "Feature A"})
    expected_targets_1 = y.iloc[[0, 1]]

    assert len(result.get_row_ids()) == 2
    assert result.get_examples(row_id=0).shape[0] == 2
    assert_frame_equal(result.get_examples(row_id=0), expected_examples_1)
    assert_series_equal(result.get_targets(row_id=0), expected_targets_1)

    expected_examples_2 = (x.iloc[[2, 1], :] + 1).rename(columns={"A": "Feature A"})
    expected_targets_2 = y.iloc[[2, 1]]

    assert len(result.get_row_ids()) == 2
    assert result.get_examples(row_id=1).shape[0] == 2
    assert_frame_equal(result.get_examples(row_id=1), expected_examples_2)
    assert_series_equal(result.get_targets(row_id=1), expected_targets_2)


def test_produce_with_standardize(dummy_model):
    x = pd.DataFrame([[1, 100], [3, 100], [1, 200], [3, 300]])
    y = pd.Series([1, 2, 3, 4])
    explainer = SimilarExamples(
        model=dummy_model,
        x_train_orig=x,
        y_train=y,
        fit_on_init=True,
        standardize=True,
    )
    result = explainer.produce(pd.DataFrame([[1, 100]]), n=2)
    expected_examples = x.iloc[[0, 2], :]
    expected_targets = y.iloc[[0, 2]]

    assert len(result.get_row_ids()) == 1
    assert result.get_examples(row_id=0).shape[0] == 2

    assert_frame_equal(result.get_examples(), expected_examples)
    assert_series_equal(result.get_targets(), expected_targets)
