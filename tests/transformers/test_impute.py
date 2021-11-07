import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from pyreal.transformers import MultiTypeImputer


def test_fit_transform_multitype_imputer():
    imputer = MultiTypeImputer()
    x = pd.DataFrame([[3, 1, np.nan, 'a', '+'],
                      [np.nan, 3, 4, 'a', '-'],
                      [6, 7, 2, np.nan, '-'],
                      [3, 9, 6, 'b', '+']], columns=["A", "B", "C", "D", "E"])
    expected_result = pd.DataFrame([[3, 1, 4, 'a', '+'],
                                    [4, 3, 4, 'a', '-'],
                                    [6, 7, 2, 'a', '-'],
                                    [3, 9, 6, 'b', '+']], columns=["A", "B", "C", "D", "E"])
    result = imputer.fit_transform(x)
    assert_frame_equal(expected_result, result, check_dtype=False)
