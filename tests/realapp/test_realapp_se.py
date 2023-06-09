import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

from pyreal import RealApp


def test_produce_similar_examples(regression_one_hot_with_interpret):
    x = pd.DataFrame([[2, 0, 0], [6, 6, 6], [6, 7, 8], [2, 0, 1]], columns=["A", "B", "C"])
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
    assert_frame_equal(explanation["id1"]["X"], x.iloc[[2, 1], :] + 1)
    assert_series_equal(explanation["id1"]["y"], y.iloc[[2, 1]])

    assert "id2" in explanation
    assert_frame_equal(explanation["id2"]["X"], x.iloc[[3, 0], :] + 1)
    assert_series_equal(explanation["id2"]["y"], y.iloc[[3, 0]])
