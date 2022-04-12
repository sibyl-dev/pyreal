import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

from pyreal.transformers import MultiTypeImputer


# TODO: Issue  # 100. Replace an np.nan with None
def test_fit_transform_multitype_imputer():
    imputer = MultiTypeImputer()
    x = pd.DataFrame(
        [
            [3, 1, np.nan, "a", "+"],
            [np.nan, 3, 4, "a", "-"],
            [6, 7, 2, np.nan, "-"],
            [3, 9, 6, "b", "+"],
        ],
        columns=["A", "B", "C", "D", "E"],
    )
    expected_result = pd.DataFrame(
        [[3, 1, 4, "a", "+"], [4, 3, 4, "a", "-"], [6, 7, 2, "a", "-"], [3, 9, 6, "b", "+"]],
        columns=["A", "B", "C", "D", "E"],
    )
    result = imputer.fit_transform(x)
    assert_frame_equal(expected_result, result, check_dtype=False)

    row = pd.Series([3, 1, np.nan, "a", "+"], index=["A", "B", "C", "D", "E"])
    expected_row = pd.Series([3, 1, 4, "a", "+"], index=["A", "B", "C", "D", "E"])

    row_result = imputer.transform(row)
    assert_series_equal(expected_row, row_result, check_dtype=False)


def test_fit_transform_multitype_imputer_cat_only():
    imputer = MultiTypeImputer()
    x = pd.DataFrame([["a", "+"], ["a", "-"], [np.nan, "-"]], columns=["D", "E"])

    expected_result = pd.DataFrame([["a", "+"], ["a", "-"], ["a", "-"]], columns=["D", "E"])
    result = imputer.fit_transform(x)
    assert_frame_equal(expected_result, result, check_dtype=False)

    row = pd.Series([np.nan, "+"], index=["D", "E"])
    expected_row = pd.Series(["a", "+"], index=["D", "E"])

    row_result = imputer.transform(row)
    assert_series_equal(expected_row, row_result, check_dtype=False)


def test_fit_transform_multitype_imputer_num_only():
    imputer = MultiTypeImputer()
    x = pd.DataFrame([[3, 1], [1, 1], [np.nan, 1]], columns=["A", "B"])

    expected_result = pd.DataFrame([[3, 1], [1, 1], [2, 1]], columns=["A", "B"])
    result = imputer.fit_transform(x)
    assert_frame_equal(expected_result, result, check_dtype=False)

    row = pd.Series([np.nan, 2], index=["A", "B"])
    expected_row = pd.Series([2, 2], index=["A", "B"])

    row_result = imputer.transform(row)
    assert_series_equal(expected_row, row_result, check_dtype=False)
