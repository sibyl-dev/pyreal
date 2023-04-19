from pyreal.transformers import BoolToIntCaster
import pandas as pd
from pandas.testing import assert_frame_equal


def test_bool_to_int_data_transform():
    x = pd.DataFrame([["A", False],
                      ["B", True],
                      [True, False]])
    x_expected = pd.DataFrame([["A", 0],
                               ["B", 1],
                               [1, 0]])

    x_transformed = BoolToIntCaster().data_transform(x)
    assert_frame_equal(x_expected, x_transformed)


def test_bool_to_int_fit():
    # no-op, ensure no break
    BoolToIntCaster().fit(pd.DataFrame([1, 2]))

