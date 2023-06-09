import pandas as pd
import numpy as np

from pyreal.explainers.example.similar_examples import SimilarExamples


def test_produce(dummy_model):
    X = pd.DataFrame([[0, 0, 0], [4, 5, 3], [0, 1, 1], [5, 5, 3]])
    y = pd.Series([0, 1, 0, 1])

    explainer = SimilarExamples(model=dummy_model, x_train_orig=X, y_train=y, fit_on_init=True)
    result = explainer.produce(pd.DataFrame([[0, 1, 0]]), n=2)
    nearest_neighbors = [[0, 0, 0], [0, 1, 1]]
    assert len(result.get_keys()) == 2
    found = [False, False]
    for item in result.get_all_examples(include_targets=True):
        assert item[1] == 0
        for i in range(len(nearest_neighbors)):
            if list(item[0]) == nearest_neighbors[i]:
                found[i] = True
    assert all(found)


def test_produce_with_transforms(regression_one_hot_with_interpret):
    x = pd.DataFrame([[2, 1, 3], [4, 3, 4], [6, 7, 10]], columns=["A", "B", "C"])
    y = pd.DataFrame([1, 2, 3])
    explainer = SimilarExamples(model=regression_one_hot_with_interpret["model"],
                                x_train_orig=x,
                                y_train=y,
                                transformers=regression_one_hot_with_interpret["transformers"],
                                fit_on_init=True)
    result = explainer.produce(pd.DataFrame([[2, 1, 4]], columns=["A", "B", "C"]), n=1)
    assert len(result.get_keys()) == 1
    np.testing.assert_array_equal(np.array(result.get_all_examples()[0]), (np.array(x.iloc[0, :])+1))
    assert result.get_target(list(result.get_keys())[0]) == y.iloc[0].squeeze()

