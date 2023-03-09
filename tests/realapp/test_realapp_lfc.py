import pandas as pd
from pyreal import RealApp
from pyreal.realapp.realapp import _get_average_or_mode
import numpy as np


def test_average_or_mode():
    mix = pd.DataFrame([[1, "a", 2, "a"],
                        [1, "a", 4, "a"],
                        [1, "b", 3, "a"]])
    expected = [1, "a", 3, "a"]
    result = _get_average_or_mode(mix)
    for i in range(len(expected)):
        assert expected[i] == result[i]

    mode_only = pd.DataFrame([["a", "a", "b", "d"],
                              ["b", "a", "b", "d"],
                              ["a", "b", "d", "d"]])
    expected = ["a", "a", "b", "d"]
    result = _get_average_or_mode(mode_only)
    for i in range(len(expected)):
        assert expected[i] == result[i]

    mean_only = pd.DataFrame([[1, 0, 2, 2],
                              [1, 5, 3, 3],
                              [1, -5, 4, 4]])
    expected = [1, 0, 3, 3]
    result = _get_average_or_mode(mean_only)
    for i in range(len(expected)):
        assert expected[i] == result[i]


def test_produce_local_feature_contributions(regression_no_transforms):
    realApp = RealApp(
        regression_no_transforms["model"],
        regression_no_transforms["x"],
        transformers=regression_no_transforms["transformers"],
    )

    x_one_dim = pd.DataFrame([[2, 10, 10]], columns=["A", "B", "C"])
    x_multi_dim = pd.DataFrame([[2, 1, 1], [4, 2, 3]], columns=["A", "B", "C"])

    expected = np.mean(regression_no_transforms["y"])[0]
    explanation = realApp.produce_local_feature_contributions(x_one_dim)

    print(explanation)

    '''assert x_one_dim.shape == contributions.shape
    assert contributions.iloc[0, 0] == x_one_dim.iloc[0, 0] - expected
    assert contributions.iloc[0, 1] == 0
    assert contributions.iloc[0, 2] == 0

    contributions = explainer.produce(x_multi_dim).get()
    assert x_multi_dim.shape == contributions.shape
    assert contributions.iloc[0, 0] == x_multi_dim.iloc[0, 0] - expected
    assert contributions.iloc[1, 0] == x_multi_dim.iloc[1, 0] - expected
    assert (contributions.iloc[:, 1] == 0).all()
    assert (contributions.iloc[:, 2] == 0).all()'''
