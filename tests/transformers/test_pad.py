import numpy as np

from pyreal.transformers import TimeSeriesPadder

X = np.array([[1, 1], [1], [1, 1, 1]])


def test_fit_transform_pad_to_length(transformer_test_data):
    transformer = TimeSeriesPadder(value=0, length=5)
    x_padded = np.array([[1, 1, 0, 0, 0], [1, 0, 0, 0, 0], [1, 1, 1, 0, 0]])
    transformer.fit(transformer_test_data["x"])
    x_trans = transformer.transform(X)

    np.testing.assert_array_equal(x_padded, x_trans)


def test_fit_transform_pad_to_longest_fit(transformer_test_data):
    transformer = TimeSeriesPadder(value=0)
    x_padded = np.array([[1, 1, 0, 0], [1, 0, 0, 0], [1, 1, 1, 0]])
    transformer.fit(np.array(transformer_test_data["x"]))  # longest is length 4
    x_trans = transformer.transform(X)

    np.testing.assert_array_equal(x_padded, x_trans)

    transformer.fit(transformer_test_data["x"])  # longest is length 4
    x_trans = transformer.transform(X)

    np.testing.assert_array_equal(x_padded, x_trans)


def test_fit_transform_pad_to_longest_transform(transformer_test_data):
    transformer = TimeSeriesPadder(value=0)
    x_padded = np.array([[1, 1, 0], [1, 0, 0], [1, 1, 1]])
    x_trans = transformer.transform(X)

    np.testing.assert_array_equal(x_padded, x_trans)


def test_fit_transform_cut_to_length(transformer_test_data):
    transformer = TimeSeriesPadder(value=0, length=1)
    x_padded = np.array([[1], [1], [1]])
    x_trans = transformer.transform(X)

    np.testing.assert_array_equal(x_padded, x_trans)


def test_fit_transform_cut_and_pad_to_length(transformer_test_data):
    transformer = TimeSeriesPadder(value=0, length=2)
    x_padded = np.array([[1, 1], [1, 0], [1, 1]])
    x_trans = transformer.transform(X)

    np.testing.assert_array_equal(x_padded, x_trans)


def test_fit_transform_cut_and_pad_to_length_fit(transformer_test_data):
    transformer = TimeSeriesPadder(value=0)
    x_padded = np.array([[1, 1], [1, 0], [1, 1]])
    transformer.fit(np.array([[1, 1]]))
    x_trans = transformer.transform(X)

    np.testing.assert_array_equal(x_padded, x_trans)


def test_fit_transform_fit_with_length_defined(transformer_test_data):
    transformer = TimeSeriesPadder(value=0, length=2)
    x_padded = np.array([[1, 1], [1, 0], [1, 1]])
    transformer.fit(np.array(transformer_test_data["x"]))  # shouldn't override set length
    x_trans = transformer.transform(X)

    np.testing.assert_array_equal(x_padded, x_trans)
