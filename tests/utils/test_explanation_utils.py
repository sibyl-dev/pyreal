import pandas as pd
from pandas.testing import assert_frame_equal

from pyreal.utils import explanation_utils


def test_get_top_contributors_contributions():
    exp = pd.DataFrame(
        [["A", 1, 5, 0], ["B", 2, 3, 0], ["C", 3, 0, 0], ["D", 4, -2, 0], ["E", 5, -6, 0]],
        columns=["Feature Name", "Feature Value", "Contribution", "Average/Mode"],
    )
    absolute = explanation_utils.get_top_contributors(exp, n=2, select_by="absolute")
    expected_absolute = exp.iloc[[4, 0], :]
    assert_frame_equal(absolute, expected_absolute)

    max = explanation_utils.get_top_contributors(exp, n=2, select_by="max")
    expected_max = exp.iloc[[0, 1], :]
    assert_frame_equal(max, expected_max)

    min = explanation_utils.get_top_contributors(exp, n=2, select_by="min")
    expected_min = exp.iloc[[4, 3], :]
    assert_frame_equal(min, expected_min)

    min_3 = explanation_utils.get_top_contributors(exp, n=3, select_by="min")
    expected_min_3 = exp.iloc[[4, 3, 2], :]
    assert_frame_equal(min_3, expected_min_3)


def test_get_top_contributors_importance():
    exp = pd.DataFrame(
        [["A", 5], ["B", 3], ["C", 0], ["D", -2], ["E", -6]],
        columns=[
            "Feature Name",
            "Importance",
        ],
    )
    absolute = explanation_utils.get_top_contributors(exp, n=2, select_by="absolute")
    expected_absolute = exp.iloc[[4, 0], :]
    assert_frame_equal(absolute, expected_absolute)

    max = explanation_utils.get_top_contributors(exp, n=2, select_by="max")
    expected_max = exp.iloc[[0, 1], :]
    assert_frame_equal(max, expected_max)

    min = explanation_utils.get_top_contributors(exp, n=2, select_by="min")
    expected_min = exp.iloc[[4, 3], :]
    assert_frame_equal(min, expected_min)

    min_3 = explanation_utils.get_top_contributors(exp, n=3, select_by="min")
    expected_min_3 = exp.iloc[[4, 3, 2], :]
    assert_frame_equal(min_3, expected_min_3)
