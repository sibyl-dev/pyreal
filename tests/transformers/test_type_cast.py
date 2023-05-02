import pandas as pd
from pandas.testing import assert_frame_equal

from pyreal.transformers import BoolToIntCaster


def test_bool_to_int_data_transform():
    x = pd.DataFrame([["A", False], ["B", True], [True, False]])
    x_expected = pd.DataFrame([["A", 0], ["B", 1], [1, 0]])

    x_transformed = BoolToIntCaster().data_transform(x)
    assert_frame_equal(x_expected, x_transformed)


def test_fit_returns_self():
    transformer = BoolToIntCaster()
    result = transformer.fit(None)
    assert result == transformer
