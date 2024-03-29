import pandas as pd
import pytest

from pyreal.transformers.generic_transformer import Transformer


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
    assert result is transformer


def test_from_transform_function_pandas():
    def transform_func(x):
        return x + 1

    transformer = Transformer.from_transform_function(transform_func)
    data_to_transform = pd.DataFrame([[1, 1, 1], [2, 1, 1]], columns=["A", "B", "C"])
    expected = pd.DataFrame([[2, 2, 2], [3, 2, 2]], columns=["A", "B", "C"])
    result = transformer.transform(data_to_transform)
    pd.testing.assert_frame_equal(result, expected)


def test_from_transform_function_numpy():
    def transform_func(x):
        return x.to_numpy() + 1

    transformer = Transformer.from_transform_function(transform_func)
    data_to_transform = pd.DataFrame([[1, 1, 1], [2, 1, 1]], columns=["A", "B", "C"])
    expected = pd.DataFrame([[2, 2, 2], [3, 2, 2]], columns=["A", "B", "C"])
    result = transformer.transform(data_to_transform)
    pd.testing.assert_frame_equal(result, expected)


def test_wrapping_transformer_without_fit():
    class TransformerWithoutFit:
        def transform(self, x):
            return x + 1

    transformer = Transformer(wrapped_transformer=TransformerWithoutFit())
    data_to_transform = pd.DataFrame([[1, 1, 1], [2, 1, 1]], columns=["A", "B", "C"])
    fit_data = pd.DataFrame([[1, 1, 1]], columns=["A", "B", "C"])
    expected = pd.DataFrame([[2, 2, 2], [3, 2, 2]], columns=["A", "B", "C"])
    fit_result = transformer.fit(fit_data)
    assert fit_result is transformer
    assert transformer.fitted
    result = transformer.transform(data_to_transform)
    pd.testing.assert_frame_equal(result, expected)
