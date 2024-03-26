import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from pyreal.transformers import (
    Mappings,
    MappingsOneHotDecoder,
    MappingsOneHotEncoder,
    OneHotEncoder,
)


def test_fit_transform_one_hot_encoder(transformer_test_data):
    ohe_transformer = OneHotEncoder(columns=pd.Index(transformer_test_data["columns"]))
    ohe_transformer.fit(transformer_test_data["x"])
    transformed_x = ohe_transformer.transform(transformer_test_data["x"])
    expected_transformed_x = pd.DataFrame(
        [[1, 9, 0, 1, 0, 1, 0, 0], [3, 0, 0, 0, 1, 0, 1, 0], [7, 2, 1, 0, 0, 0, 0, 1]],
        columns=["B", "D", "C_2", "C_3", "C_4", "A_2", "A_4", "A_6"],
    )
    assert_frame_equal(transformed_x, expected_transformed_x, check_dtype=False)


def test_transform_one_hot_encoder(transformer_test_data):
    ohe_transformer = OneHotEncoder(columns=transformer_test_data["columns"])
    ohe_transformer.fit(transformer_test_data["x"])
    test_x = pd.DataFrame([[6, 0, 2, 1], [4, 1, 3, 8], [2, 0, 4, 3]], columns=["A", "B", "C", "D"])
    transformed_x = ohe_transformer.transform(test_x)
    expected_transformed_x = pd.DataFrame(
        [[0, 1, 1, 0, 0, 0, 0, 1], [1, 8, 0, 1, 0, 0, 1, 0], [0, 3, 0, 0, 1, 1, 0, 0]],
        columns=["B", "D", "C_2", "C_3", "C_4", "A_2", "A_4", "A_6"],
    )
    assert_frame_equal(transformed_x, expected_transformed_x, check_dtype=False)


def test_transform_all_columns_one_hot_encoder(transformer_test_data):
    fs_transformer = OneHotEncoder()
    transformed_x = fs_transformer.fit_transform(transformer_test_data["x"])
    expected_transformed_x = pd.DataFrame(
        [
            [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
            [0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0],
        ],
        columns=[
            "A_2",
            "A_4",
            "A_6",
            "B_1",
            "B_3",
            "B_7",
            "C_2",
            "C_3",
            "C_4",
            "D_0",
            "D_2",
            "D_9",
        ],
    )
    assert_frame_equal(transformed_x, expected_transformed_x, check_dtype=False)


def test_inverse_transform_one_hot_encoder(transformer_test_data):
    ohe_transformer = OneHotEncoder(columns=transformer_test_data["columns"])
    ohe_transformer.fit(transformer_test_data["x"])
    test_x = pd.DataFrame([[6, 0, 2, 1], [4, 1, 3, 8], [2, 0, 4, 3]], columns=["A", "B", "C", "D"])
    transformed_x = ohe_transformer.transform(test_x)
    identity_x = ohe_transformer.inverse_transform(transformed_x)
    assert_frame_equal(identity_x, test_x, check_dtype=False)


categorical_to_one_hot = {
    "A": {"A_a": "a", "A_b": "b"},
    "B": {"B_a": "a", "B_b": "b", "B_c": "c"},
}
one_hot_to_categorical = {
    "A_a": ("A", "a"),
    "A_b": ("A", "b"),
    "B_a": ("B", "a"),
    "B_b": ("B", "b"),
    "B_c": ("B", "c"),
}
dataframe = pd.DataFrame(
    [
        ["A_a", "A", "a"],
        ["A_b", "A", "b"],
        ["B_a", "B", "a"],
        ["B_b", "B", "b"],
        ["B_c", "B", "c"],
    ],
    columns=["one_hot_encoded", "categorical", "value"],
)
mappings_ctoh = Mappings.generate_mappings(categorical_to_one_hot=categorical_to_one_hot)
mappings_ohtc = Mappings.generate_mappings(one_hot_to_categorical=one_hot_to_categorical)
mappings_df = Mappings.generate_mappings(dataframe=dataframe)
mappings_choices = [mappings_ctoh, mappings_ohtc, mappings_df]


@pytest.mark.parametrize("mappings", mappings_choices)
def test_mappings_encode_decode(mappings):
    mappings_ohe = MappingsOneHotEncoder(mappings)

    x = pd.DataFrame([["a", "b", 10, "f"], ["b", "c", 11, "d"]], columns=["A", "B", "C", "D"])

    x_expected = pd.DataFrame(
        [
            [True, False, False, True, False, 10, "f"],
            [False, True, False, False, True, 11, "d"],
        ],
        columns=["A_a", "A_b", "B_a", "B_b", "B_c", "C", "D"],
    )
    x_encoded = mappings_ohe.transform(x)
    assert_frame_equal(x_encoded, x_expected)

    mappings_ohd = MappingsOneHotDecoder(mappings)
    x_decoded = mappings_ohd.transform(x_encoded)

    assert_frame_equal(x_decoded, x)


@pytest.mark.parametrize("mappings", mappings_choices)
def test_mappings_inverse_transform(mappings):
    mappings_ohe = MappingsOneHotEncoder(mappings)

    x = pd.DataFrame([["a", "b", 10, "f"], ["b", "c", 11, "d"]], columns=["A", "B", "C", "D"])
    x_encoded = mappings_ohe.transform(x)
    x_identity = mappings_ohe.inverse_transform(x_encoded)
    assert_frame_equal(x_identity, x)

    mappings_ohd = MappingsOneHotDecoder(mappings)
    x_encoded_identity = mappings_ohd.inverse_transform(x)

    assert_frame_equal(x_encoded_identity, x_encoded)


def test_fit_returns_self():
    for transformer in [
        MappingsOneHotDecoder(mappings_ctoh),
        MappingsOneHotEncoder(mappings_ctoh),
        OneHotEncoder(columns=[]),
    ]:
        result = transformer.fit(pd.DataFrame([0, 1]))
        assert result == transformer
