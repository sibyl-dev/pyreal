import pandas as pd
from pandas.testing import assert_series_equal

from pyreal import RealApp


def test_produce_similar_examples(regression_one_hot_with_interpret):
    x = pd.DataFrame([[0, 0, 0], [6, 6, 6], [8, 7, 8], [1, 0, 1]], columns=["A", "B", "C"])
    y = pd.Series([1, 0, 0, 1])
    realApp = RealApp(
        regression_one_hot_with_interpret["model"],
        X_train_orig=x,
        y_train=y,
        transformers=regression_one_hot_with_interpret["transformers"],
        id_column="ID",
    )

    explanation = realApp.produce_similar_examples(
        pd.DataFrame([["id1", 6, 7, 9], ["id2", 2, 1, 1]], columns=["ID", "A", "B", "C"]), n=2
    )
    assert "id1" in explanation
    assert explanation["id1"]["X"] == pd.DataFrame((x.iloc[[0, 3], :] + 1))
