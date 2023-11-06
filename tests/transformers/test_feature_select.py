import numpy as np
import pandas as pd
from explanation_types.feature_based import FeatureBased
from pandas.testing import assert_frame_equal

from pyreal.transformers import ColumnDropTransformer, FeatureSelectTransformer

X = pd.DataFrame([[2, 1, 3, 9], [4, 3, 4, 0], [6, 7, 2, 2]], columns=["A", "B", "C", "D"])
COLUMNS = ["C", "A"]


def test_fit_feature_select_transformer(transformer_test_data):
    fs_transformer = FeatureSelectTransformer(columns=transformer_test_data["columns"])
    fs_transformer.fit(transformer_test_data["x"])
    assert (fs_transformer.dropped_columns == ["D", "B"]) or (
        fs_transformer.dropped_columns == ["B", "D"]
    )


def test_transform_feature_select_transformer(transformer_test_data):
    fs_transformer = FeatureSelectTransformer(columns=transformer_test_data["columns"])
    fs_transformer.fit(transformer_test_data["x"])
    transformed_x = fs_transformer.transform(transformer_test_data["x"])
    expected_transformed_x = pd.DataFrame([[3, 2], [4, 4], [2, 6]], columns=["C", "A"])
    assert_frame_equal(transformed_x, expected_transformed_x)


def test_transform_column_drop_transformer(transformer_test_data):
    cd_transformer = ColumnDropTransformer(columns=transformer_test_data["columns"])
    transformed_x = cd_transformer.transform(transformer_test_data["x"])
    expected_transformed_x = pd.DataFrame([[1, 9], [3, 0], [7, 2]], columns=["B", "D"])
    assert_frame_equal(transformed_x, expected_transformed_x)


def test_fit_transform_feature_select_transformer_other_formats(transformer_test_data):
    for columns in ["A", np.array(["A"]), pd.Index(["A"])]:
        fs_transformer = FeatureSelectTransformer(columns="A")
        fs_transformer.fit(transformer_test_data["x"])
        assert set(fs_transformer.dropped_columns) == {"B", "C", "D"}
        transformed_x = fs_transformer.transform(transformer_test_data["x"])
        expected_transformed_x = pd.DataFrame([[2], [4], [6]], columns=["A"])
        assert_frame_equal(transformed_x, expected_transformed_x)


def test_transform_explanation_feature_select(transformer_test_data):
    fs_transformer = FeatureSelectTransformer(columns=COLUMNS)
    explanation = FeatureBased(
        pd.DataFrame([[1, 2, 3, 4], [1, 2, 3, 4]], columns=["A", "B", "C", "D"])
    )
    expected_explanation = pd.DataFrame([[3, 1], [3, 1]], columns=["C", "A"])
    fs_transformer.fit(X)

    trans_exp = fs_transformer.transform_explanation(explanation)

    assert_frame_equal(trans_exp.get(), expected_explanation)


def test_inverse_transform_explanation_feature_select(transformer_test_data):
    fs_transformer = FeatureSelectTransformer(columns=COLUMNS)
    explanation = FeatureBased(pd.DataFrame([[3, 1], [3, 1]], columns=["C", "A"]))
    expected_explanation = pd.DataFrame([[1, 0, 3, 0], [1, 0, 3, 0]], columns=["A", "B", "C", "D"])
    fs_transformer.fit(X)

    trans_exp = fs_transformer.inverse_transform_explanation(explanation)

    assert_frame_equal(trans_exp.get(), expected_explanation)


def test_fit_returns_self():
    transformer = FeatureSelectTransformer(columns=[])
    result = transformer.fit(pd.DataFrame([0, 1]))
    assert result == transformer
