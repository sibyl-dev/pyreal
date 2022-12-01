import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

from pyreal.transformers import MultiTypeImputer


def test_fit_transform_multitype_imputer():
    imputer = MultiTypeImputer()
    x = pd.DataFrame(
        [
            [3, 1, None, "a", "+"],
            [np.nan, 3, 4, "a", "-"],
            [6, 7, 2, np.nan, "-"],
            [3, 9, 6, "b", "+"],
        ],
        columns=["A", "B", "C", "D", "E"],
    )
    x = x.astype({"A": "Int64", "B": "Int64", "C": "Int64"})

    expected_result = pd.DataFrame(
        [[3, 1, 4, "a", "+"], [4, 3, 4, "a", "-"], [6, 7, 2, "a", "-"], [3, 9, 6, "b", "+"]],
        columns=["A", "B", "C", "D", "E"],
    )
    expected_result = expected_result.astype({"A": "Int64", "B": "Int64", "C": "Int64"})
    result = imputer.fit_transform(x)

    assert_frame_equal(expected_result, result)

    row = pd.Series([3, 1, np.nan, "a", "+"], index=["A", "B", "C", "D", "E"])
    expected_row = pd.Series([3, 1, 4, "a", "+"], index=["A", "B", "C", "D", "E"])

    row_result = imputer.transform(row)
    assert_series_equal(expected_row, row_result)


def test_fit_transform_multitype_imputer_cat_only():
    imputer = MultiTypeImputer()
    x = pd.DataFrame([["a", "+"], ["a", "-"], [None, "-"]], columns=["D", "E"])

    expected_result = pd.DataFrame([["a", "+"], ["a", "-"], ["a", "-"]], columns=["D", "E"])
    result = imputer.fit_transform(x)
    assert_frame_equal(expected_result, result)

    row = pd.Series([None, "+"], index=["D", "E"])
    expected_row = pd.Series(["a", "+"], index=["D", "E"])

    row_result = imputer.transform(row)
    assert_series_equal(expected_row, row_result)


def test_fit_transform_multitype_imputer_num_only():
    imputer = MultiTypeImputer()
    x = pd.DataFrame([[3.0, 1.0], [1.0, 1.0], [np.nan, 1.0]], columns=["A", "B"])

    expected_result = pd.DataFrame([[3.0, 1.0], [1.0, 1.0], [2.0, 1.0]], columns=["A", "B"])
    result = imputer.fit_transform(x)
    assert_frame_equal(expected_result, result)

    row = pd.Series([np.nan, 2.0], index=["A", "B"])
    expected_row = pd.Series([2.0, 2.0], index=["A", "B"])

    row_result = imputer.transform(row)
    assert_series_equal(expected_row, row_result)


def test_fit_transform_multitype_keep_int_type():
    imputer = MultiTypeImputer()
    x = pd.DataFrame(
        [
            [3.2, 1],
            [np.nan, 3],
            [6.1, 7],
            [3.5, 4],
        ],
        columns=["A", "B"],
    )

    expected_result = pd.DataFrame(
        [[3.2, 1], [4.266, 3], [6.1, 7], [3.5, 4]],
        columns=["A", "B"],
    )
    result = imputer.fit_transform(x)
    assert_frame_equal(expected_result, result, check_exact=False, atol=1e-3)


def test_fit_transform_column_parameter():
    x = pd.DataFrame(
        [
            [3.2, 1, 1.0],
            [np.nan, 3, 1.0],
            [6.1, np.nan, 1.0],
            [3.5, np.nan, np.nan],
        ],
        columns=["A", "B", "C"],
    )

    imputer = MultiTypeImputer(columns="A")
    expected_result = pd.DataFrame(
        [[3.2, 1, 1.0], [4.266, 3, 1.0], [6.1, np.nan, 1.0], [3.5, np.nan, np.nan]],
        columns=["A", "B", "C"],
    )

    result = imputer.fit_transform(x)
    assert_frame_equal(expected_result, result, check_exact=False, atol=1e-3)

    imputer = MultiTypeImputer(columns=["A", "C"])
    expected_result = pd.DataFrame(
        [[3.2, 1, 1.0], [4.266, 3, 1.0], [6.1, np.nan, 1.0], [3.5, np.nan, 1.0]],
        columns=["A", "B", "C"],
    )

    result = imputer.fit_transform(x)
    assert_frame_equal(expected_result, result, check_exact=False, atol=1e-3)


def test_fit_transform_multitype_columns_parameter():
    x = pd.DataFrame(
        [
            [3, 1, None, "a", "+"],
            [np.nan, 3, 4, "a", "-"],
            [6, 7, 2, np.nan, "-"],
            [3, 9, 6, "b", "+"],
        ],
        columns=["A", "B", "C", "D", "E"],
    )
    x = x.astype({"A": "Int64", "B": "Int64", "C": "Int64"})

    imputer = MultiTypeImputer(columns="A")
    expected_result = pd.DataFrame(
        [
            [3, 1, None, "a", "+"],
            [4, 3, 4, "a", "-"],
            [6, 7, 2, np.nan, "-"],
            [3, 9, 6, "b", "+"],
        ],
        columns=["A", "B", "C", "D", "E"],
    )
    expected_result = expected_result.astype({"A": "Int64", "B": "Int64", "C": "Int64"})
    result = imputer.fit_transform(x)

    assert_frame_equal(expected_result, result)

    imputer = MultiTypeImputer(columns=["D"])
    expected_result = pd.DataFrame(
        [
            [3, 1, None, "a", "+"],
            [np.nan, 3, 4, "a", "-"],
            [6, 7, 2, "a", "-"],
            [3, 9, 6, "b", "+"],
        ],
        columns=["A", "B", "C", "D", "E"],
    )
    expected_result = expected_result.astype({"A": "Int64", "B": "Int64", "C": "Int64"})
    result = imputer.fit_transform(x)

    assert_frame_equal(expected_result, result)

    imputer = MultiTypeImputer(columns=["A", "D"])
    expected_result = pd.DataFrame(
        [
            [3, 1, None, "a", "+"],
            [4, 3, 4, "a", "-"],
            [6, 7, 2, "a", "-"],
            [3, 9, 6, "b", "+"],
        ],
        columns=["A", "B", "C", "D", "E"],
    )
    expected_result = expected_result.astype({"A": "Int64", "B": "Int64", "C": "Int64"})
    result = imputer.fit_transform(x)

    assert_frame_equal(expected_result, result)
