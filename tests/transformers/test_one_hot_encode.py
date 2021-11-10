import pandas as pd
from pandas.testing import assert_frame_equal

from pyreal.transformers import OneHotEncoder


def test_fit_transform_one_hot_encoder(transformer_test_data):
    ohe_transformer = OneHotEncoder(columns=transformer_test_data["columns"])
    ohe_transformer.fit(transformer_test_data["x"])
    transformed_x = ohe_transformer.transform(transformer_test_data["x"])
    expected_transformed_x = pd.DataFrame([[1, 9, 0, 1, 0, 1, 0, 0],
                                           [3, 0, 0, 0, 1, 0, 1, 0],
                                           [7, 2, 1, 0, 0, 0, 0, 1]],
                                          columns=["B", "D", "C_2", "C_3",
                                                   "C_4", "A_2", "A_4", "A_6"])
    assert_frame_equal(transformed_x, expected_transformed_x, check_dtype=False)


def test_transform_one_hot_encoder(transformer_test_data):
    ohe_transformer = OneHotEncoder(columns=transformer_test_data["columns"])
    ohe_transformer.fit(transformer_test_data["x"])
    test_x = pd.DataFrame([[6, 0, 2, 1],
                           [4, 1, 3, 8],
                           [2, 0, 4, 3]], columns=["A", "B", "C", "D"])
    transformed_x = ohe_transformer.transform(test_x)
    expected_transformed_x = pd.DataFrame([[0, 1, 1, 0, 0, 0, 0, 1],
                                           [1, 8, 0, 1, 0, 0, 1, 0],
                                           [0, 3, 0, 0, 1, 1, 0, 0]],
                                          columns=["B", "D", "C_2", "C_3", "C_4",
                                                   "A_2", "A_4", "A_6"])
    assert_frame_equal(transformed_x, expected_transformed_x, check_dtype=False)


def test_transform_all_columns_one_hot_encoder(transformer_test_data):
    fs_transformer = OneHotEncoder()
    transformed_x = fs_transformer.fit_transform(transformer_test_data["x"])
    expected_transformed_x = pd.DataFrame([[1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
                                           [0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0],
                                           [0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0]],
                                          columns=["A_2", "A_4", "A_6", "B_1", "B_3", "B_7",
                                                   "C_2", "C_3", "C_4", "D_0", "D_2", "D_9"])
    assert_frame_equal(transformed_x, expected_transformed_x, check_dtype=False)
