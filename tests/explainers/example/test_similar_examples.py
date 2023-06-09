import pandas as pd
from pandas.testing import assert_series_equal, assert_frame_equal

from pyreal.explainers.example.similar_examples import SimilarExamples


def test_produce(dummy_model):
    X = pd.DataFrame([[1, 1, 1], [4, 5, 3], [0, 0, 0], [5, 5, 3]])
    y = pd.Series([0, 1, 0, 1])

    explainer = SimilarExamples(model=dummy_model, x_train_orig=X, y_train=y, fit_on_init=True)
    result = explainer.produce(pd.DataFrame([[0, 1, 0]]), n=2)
    expected_examples = pd.DataFrame([[0, 0, 0], [1, 1, 1]])
    expected_targets = pd.Series([0, 0])
    assert len(result.get_row_ids()) == 1
    assert result.get_examples(row_id=0).shape[0] == 2
    assert_frame_equal(result.get_examples().reset_index(drop=True), expected_examples)
    assert_series_equal(result.get_targets().reset_index(drop=True), expected_targets)


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
    assert_frame_equal(result.get_examples().reset_index(drop=True), expected_examples)
    print(result.get_targets())
    print(expected_targets)
    assert_series_equal(result.get_targets().reset_index(drop=True), expected_targets)


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
    expected_examples_1 = pd.DataFrame((x.iloc[[0, 1], :] + 1)).rename(columns={"A": "Feature A"})
    expected_targets_1 = pd.Series([y.iloc[[0, 1]]]).squeeze()

    assert len(result.get_row_ids()) == 2
    assert result.get_examples(row_id=0).shape[0] == 2
    assert_frame_equal(result.get_examples(row_id=0).reset_index(drop=True), expected_examples_1)
    assert_series_equal(result.get_targets().reset_index(drop=True), expected_targets_1)
