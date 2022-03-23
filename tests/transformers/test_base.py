import pandas as pd
from pandas.testing import assert_frame_equal

from pyreal.transformers import (
    FeatureSelectTransformer,
    OneHotEncoder,
    base,
    fit_transformers,
    run_transformers,
)


def test_fit_run_transformers():
    x_orig = pd.DataFrame([
        [1, 1, 1],
        [2, 2, 2],
        [3, 3, 3]
    ], columns=["A", "B", "C"])

    ohe_transformer = OneHotEncoder(columns=["A"])
    fs_transformer = FeatureSelectTransformer(columns=["A_1", "A_2", "C"])

    transformers = [ohe_transformer, fs_transformer]

    expected_result = pd.DataFrame([
        [1, 0, 1],
        [0, 1, 2],
        [0, 0, 3]
    ], columns=["A_1", "A_2", "C"])

    result_fit = fit_transformers(transformers, x_orig)
    assert_frame_equal(result_fit, expected_result, check_dtype=False)

    result_run = run_transformers(transformers, x_orig)
    assert_frame_equal(result_run, expected_result, check_dtype=False)


def test_display_missing_transform_info():
    # Assert no errors
    base._display_missing_transform_info("A", "B")
    base._display_missing_transform_info_inverse("A", "B")
