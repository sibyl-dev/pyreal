from sklearn.preprocessing import MinMaxScaler as SklearnMinMaxScaler
from sklearn.preprocessing import Normalizer as SklearnNormalizer
from sklearn.preprocessing import StandardScaler as SklearnStandardScaler

from pyreal.transformers import Transformer
from pyreal.transformers.wrappers import DataFrameWrapper


class MinMaxScaler:
    """

    Directly implements a sklearn MinMaxScaler into Pyreal.
    Initializes a Transformer.

    """

    def __init__(self, feature_range=(0, 1), *, clip=False):
        """
        Initialize a wrapped transformer and DataFrameWrapper, then wrap the DataFrameWrapper

        Args:
            feature_range (tuple (min, max), default=(0, 1)):
                Desired range of transformed data.
            clip (bool, default=False):
                Set to True to clip transformed values of held-out data
                to provided feature range.
        """
        self.data_frame_wrapper = DataFrameWrapper(
            SklearnMinMaxScaler(feature_range, copy=True, clip=clip)
        )

        # attributes
        self.min_ = None
        self.data_min_ = None
        self.data_max_ = None
        self.data_range_ = None

    def fit(self, X, y=None):
        """computes per-feature min & max (self.data_min_, self.data_max_)

        Args:
            X (DataFrame): represents an array to be fitted.
            y (array): target values. Defaults to None.

        Returns:
            fitted Transformer
        """

        ret = self.data_frame_wrapper.fit(X, y=y)
        self.min_ = self.data_frame_wrapper.wrapped_transformer.min_
        self.data_min_ = self.data_frame_wrapper.wrapped_transformer.data_min_
        self.data_max_ = self.data_frame_wrapper.wrapped_transformer.data_max_
        self.data_range_ = self.data_frame_wrapper.wrapped_transformer.data_range_
        return ret

    def fit_transform(self, X, y=None, **fit_params):
        """Fits and transforms

        Args:
            X (DataFrame): represents an array to be fitted.
            y (array): target values. Defaults to None.

        Returns:
            DataFrame: a fitted and transformed DataFrame
        """
        return self.data_frame_wrapper.fit_transform(X, y, **fit_params)

    def inverse_transform(self, X):
        """Inverse transform X

        Args:
            X (DataFrame): the dataset to be inverse transformed

        Returns:
            DataFrame: the result of inverse transforming X
        """
        return self.data_frame_wrapper.inverse_transform(X)

    def data_transform(self, X):
        """Transform a dataset

        Args:
            X (Dataframe): the dataset to be inverse transformed

        Returns:
            DataFrame: the result of transforming X
        """
        return self.data_frame_wrapper.transform(X)


class Normalizer:
    def __init__(self, norm="l2"):
        """
        Initialize a wrapped transformer and DataFrameWrapper, then wrap the DataFrameWrapper

        Args:
            norm (str, optional): The norm to use to normalize each non zero sample.
                                  If norm=’max’ is used, values will be rescaled by
                                  the maximum of the absolute values.
                                  Can take values {‘l1’, ‘l2’, ‘max’}. Defaults to 'l2'.
        """
        self.data_frame_wrapper = DataFrameWrapper(SklearnNormalizer(norm, copy=True))

    def fit(self, X, y=None):
        """Fits a dataset to the transformer

        Args:
            X (DataFrame): represents an array to be fitted.
            y (array): target values. Defaults to None.

        Returns:
            fitted Transformer
        """
        return self.data_frame_wrapper.fit(X, y=y)

    def fit_transform(self, X, y=None, **fit_params):
        """Fits and transforms

        Args:
            X (DataFrame): represents an array to be fitted.
            y (array): target values. Defaults to None.

        Returns:
            DataFrame: a fitted and transformed DataFrame
        """
        return self.data_frame_wrapper.fit_transform(X, y, **fit_params)

    def data_transform(self, X):
        """Transform a dataset

        Args:
            X (Dataframe): the dataset to be inverse transformed

        Returns:
            DataFrame: the result of transforming X
        """
        return self.data_frame_wrapper.transform(X)


class StandardScaler(Transformer):
    def __init__(self, *, with_mean=True, with_std=True):
        """
        Creates a pyreal StandardScaler, and wraps it a DataFrameWrapper,
        then wraps the DataFrameWrapper

        Args:
            with_mean (bool, optional):
                If True, center the data before scaling.
            with_std (bool, optional): If True, scale the data to unit variance
            (or equivalently, unit standard deviation).
        """
        self.data_frame_wrapper = DataFrameWrapper(
            SklearnStandardScaler(copy=True, with_mean=with_mean, with_std=with_std)
        )
        self.mean_ = None
        self.var_ = None

    def fit(self, X, y=None, sample_weight=None):
        """Fits a dataset to the transformer

        Args:
            X (DataFrame): represents an array to be fitted.
            y (array): target values. Defaults to None.
            sample_weight (array-like shape) weights for each sample. Defaults to NOne.

        Returns:
            fitted Transformer
        """
        ret = self.data_frame_wrapper.fit(X, y=y, sample_weight=sample_weight)
        self.mean_ = ret.wrapped_transformer.mean_
        self.var_ = ret.wrapped_transformer.var_
        return ret

    def data_transform(self, X):
        """Transform a dataset

        Args:
            X (Dataframe): the dataset to be inverse transformed

        Returns:
            DataFrame: the result of transforming X
        """
        return self.data_frame_wrapper.transform(X)

    def inverse_transform(self, X):
        """Inverse transform X

        Args:
            X (DataFrame): the dataset to be inverse transformed

        Returns:
            DataFrame: the result of inverse transforming X
        """
        return self.data_frame_wrapper.inverse_transform(X)
