from pyreal.explainers.example.similar_examples import SimilarExampleExplainer
import pandas as pd


def test_produce(dummy_model):
    X = pd.DataFrame([[0, 0, 0],
                      [4, 5, 3],
                      [0, 1, 1],
                      [5, 5, 3]])
    y = pd.Series([0, 1, 0, 1])

    explainer = SimilarExampleExplainer(model=dummy_model, x_train_orig=X, y_train=y, fit_on_init=True)
    print(explainer.produce(pd.DataFrame([[0, 1, 0]]), n=2).get())
