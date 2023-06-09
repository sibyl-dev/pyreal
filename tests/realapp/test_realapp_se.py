from pyreal import RealApp
import pandas as pd


def test_produce_similar_examples(regression_one_hot_with_interpret):
    x = pd.DataFrame([[2, 1, 3], [4, 3, 4], [6, 7, 10], [2, 1, 4]], columns=["A", "B", "C"])
    y = pd.Series([1, 0, 0, 1])
    realApp = RealApp(
        regression_one_hot_with_interpret["model"],
        X_train_orig=x,
        y_train=y,
        transformers=regression_one_hot_with_interpret["transformers"],
    )

    explanation = realApp.produce_similar_examples(pd.DataFrame([[2, 1, 4]], columns=["A", "B", "C"]), n=2)
    print(explanation)
