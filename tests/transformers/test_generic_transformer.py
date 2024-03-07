import pandas as pd

from pyreal.transformers.generic_transformer import Transformer

import pytest


class SomeTransformer:
    def __init__(self, numpy=False):
        self.add = None
        self.numpy = numpy

    def fit(self, x):
        self.add = x.to_numpy().sum()

    def transform(self, x):
        if self.add is None:
            raise ValueError("Transformer not fitted")
        if self.numpy:
            return x.to_numpy() + self.add
        return x + self.add


def helper(transformer, fit_data, data_to_transform, expected):
    with pytest.raises(ValueError):
        transformer.transform(data_to_transform)
    transformer.fit(fit_data)
    assert transformer.fitted
    result = transformer.transform(data_to_transform)
    pd.testing.assert_frame_equal(result, expected)


def test_transform_no_columns_numpy():
    transformer = Transformer(wrapped_transformer=SomeTransformer(numpy=True))
    fit_data = pd.DataFrame([[0, 0, 0], [1, 0, 0]], columns=["A", "B", "C"])  # add = 1
    data_to_transform = pd.DataFrame([[1, 1, 1], [2, 1, 1]], columns=["A", "B", "C"])
    expected = pd.DataFrame([[2, 2, 2], [3, 2, 2]], columns=["A", "B", "C"])
    helper(transformer, fit_data, data_to_transform, expected)


def test_transform_no_columns_dataframe():
    transformer = Transformer(wrapped_transformer=SomeTransformer())
    fit_data = pd.DataFrame([[0, 0, 0], [1, 1, 0]], columns=["A", "B", "C"])  # add = 2
    data_to_transform = pd.DataFrame([[1, 1, 1], [2, 1, 1]], columns=["A", "B", "C"])
    expected = pd.DataFrame([[3, 3, 3], [4, 3, 3]], columns=["A", "B", "C"])
    helper(transformer, fit_data, data_to_transform, expected)


def test_transform_with_columns_numpy():
    transformer = Transformer(wrapped_transformer=SomeTransformer(numpy=True), columns=["A", "B"])
    # add = 1 (only using columns A and B)
    fit_data = pd.DataFrame([[0, 0, 0], [1, 0, 2]], columns=["A", "B", "C"])
    data_to_transform = pd.DataFrame([[1, 1, 1], [2, 1, 1]], columns=["A", "B", "C"])
    expected = pd.DataFrame([[2, 2, 1], [3, 2, 1]], columns=["A", "B", "C"])
    helper(transformer, fit_data, data_to_transform, expected)


def test_transform_with_columns_dataframe():
    transformer = Transformer(wrapped_transformer=SomeTransformer(numpy=False), columns=["A", "B"])
    # add = 1 (only using columns A and B)
    fit_data = pd.DataFrame([[0, 0, 0], [1, 0, 2]], columns=["A", "B", "C"])
    data_to_transform = pd.DataFrame([[1, 1, 1], [2, 1, 1]], columns=["A", "B", "C"])
    expected = pd.DataFrame([[2, 2, 1], [3, 2, 1]], columns=["A", "B", "C"])
    helper(transformer, fit_data, data_to_transform, expected)


def test_fit_returns_self():
    transformer = Transformer(wrapped_transformer=SomeTransformer())
    result = transformer.fit(pd.DataFrame([1]))
    assert result == transformer
