from pyreal.realapp import utils
import pandas as pd
from pandas.testing import assert_frame_equal


def test_get_top_contributors():
    exp = pd.DataFrame(
        [["A", 1, 5, 0], ["B", 2, 3, 0], ["C", 3, 0, 0], ["D", 4, -2, 0], ["E", 5, -6, 0]],
        columns=["Feature Name", "Feature Value", "Contribution", "Average/Mode"],
    )
    absolute = utils.get_top_contributors(exp, n=2, select_by="absolute")
    expected_absolute = exp.iloc[[4, 0], :]
    assert_frame_equal(absolute, expected_absolute)

    max = utils.get_top_contributors(exp, n=2, select_by="max")
    expected_max = exp.iloc[[0, 1], :]
    assert_frame_equal(max, expected_max)

    min = utils.get_top_contributors(exp, n=2, select_by="min")
    expected_min = exp.iloc[[4, 3], :]
    assert_frame_equal(min, expected_min)

    min_3 = utils.get_top_contributors(exp, n=3, select_by="min")
    expected_min_3 = exp.iloc[[4, 3, 2], :]
    assert_frame_equal(min_3, expected_min_3)


