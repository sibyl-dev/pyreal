import pandas as pd
from pandas.testing import assert_frame_equal

from pyreal.transformers import ColumnDropTransformer, FeatureSelectTransformer


def test_fit_feature_select_transformer(transformer_test_data):
    fs_transformer = FeatureSelectTransformer(columns=transformer_test_data["columns"])
    fs_transformer.fit(transformer_test_data["x"])
    assert fs_transformer.dropped_columns == ["D", "B"]


def test_transform_feature_select_transformer(transformer_test_data):
    fs_transformer = FeatureSelectTransformer(columns=transformer_test_data["columns"])
    transformed_x = fs_transformer.transform(transformer_test_data["x"])
    expected_transformed_x = pd.DataFrame([[3, 2],
                                           [4, 4],
                                           [2, 6]], columns=["C", "A"])
    assert_frame_equal(transformed_x, expected_transformed_x)


def test_transform_column_drop_transformer(transformer_test_data):
    cd_transformer = ColumnDropTransformer(columns=transformer_test_data["columns"])
    transformed_x = cd_transformer.transform(transformer_test_data["x"])
    expected_transformed_x = pd.DataFrame([[1, 9],
                                           [3, 0],
                                           [7, 2]], columns=["B", "D"])
    assert_frame_equal(transformed_x, expected_transformed_x)
