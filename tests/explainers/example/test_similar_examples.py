import pandas as pd

from pyreal.explainers.example.similar_examples import SimilarExampleExplainer


def test_produce(dummy_model):
    X = pd.DataFrame([[0, 0, 0], [4, 5, 3], [0, 1, 1], [5, 5, 3]])
    y = pd.Series([0, 1, 0, 1])

    explainer = SimilarExampleExplainer(
        model=dummy_model, x_train_orig=X, y_train=y, fit_on_init=True
    )
    result = explainer.produce(pd.DataFrame([[0, 1, 0]]), n=2).get()
    nearest_neighbors = [[0, 0, 0], [0, 1, 1]]
    assert len(result) == 2
    found = [False, False]
    for item in result:
        assert result[item][1] == 0
        for i in range(len(nearest_neighbors)):
            if list(result[item][0]) == nearest_neighbors[i]:
                found[i] = True
    assert all(found)
