import sklearn
from sklearn.preprocessing import MinMaxScaler as SklearnMinMaxScaler
from sklearn.preprocessing import Normalizer as SklearnNormalizer
from sklearn.preprocessing import StandardScaler as SklearnStandardScaler

from pyreal.transformers import Transformer
from pyreal.transformers.wrappers import DataFrameWrapper


class MinMaxScaler:
    """
    basically a sklearn MinMaxScaler but maintains DataFrame type
    """

    def __init__(self, feature_range=(0, 1), *, copy=True, clip=False):
        """initialize a wrapped transformer, then make a DataFrameWrapper for it, then wrap the DataFrameWrapper

        Args:
            feature_range (tuple (min, max), default=(0, 1)): Desired range of transformed data.
            copy (bool, default=True): Set to False to perform inplace row normalization and avoid a copy (if the input is already a numpy array).
            clip (bool, default=False): Set to True to clip transformed values of held-out data to provided feature range.
        """
        self.data_frame_wrapper = DataFrameWrapper(
            SklearnMinMaxScaler(feature_range, copy=copy, clip=clip)
        )
        # self.sklearn = self.data_frame_wrapper.wrapped_transformer

        # attributes
        self.min_ = None
        self.scale_ = None
        self.data_min_ = None
        self.data_max_ = None
        self.data_range_ = None
        self.n_features_in_ = None
        self.n_samples_seen_ = None
        self.feature_names_in_ = None

    # methods

    def fit(self, X, y=None):
        # computes per-feature min & max (self.data_min_, self.data_max_)
        ret = self.data_frame_wrapper.fit(X, y=y)
        self.min_ = self.data_frame_wrapper.wrapped_transformer.min_
        self.data_min_ = self.data_frame_wrapper.wrapped_transformer.data_min_
        self.data_max_ = self.data_frame_wrapper.wrapped_transformer.data_max_
        self.data_range_ = self.data_frame_wrapper.wrapped_transformer.data_range_
        return ret

    def fit_transform(self, X, y=None, **fit_params):
        return self.data_frame_wrapper.fit_transform(X, y, **fit_params)

    def inverse_transform(self, X):
        return self.data_frame_wrapper.inverse_transform(X)

    def data_transform(self, X):
        return self.data_frame_wrapper.transform(X)


class Normalizer:
    def __init__(self, norm="l2", *, copy=True):
        """_summary_

        Args:
            norm (str, optional): The norm to use to normalize each non zero sample.
                                  If norm=’max’ is used, values will be rescaled by the maximum of the absolute values.
                                  Can take values {‘l1’, ‘l2’, ‘max’}. Defaults to 'l2'.
            copy (bool, optional): Set to False to perform inplace row normalization and avoid a copy
                                   (if the input is already a numpy array or a scipy.sparse CSR matrix). Defaults to True.
        """
        self.data_frame_wrapper = DataFrameWrapper(SklearnNormalizer(norm, copy=copy))

    # write methods
    def fit(self, X, y=None):
        return self.data_frame_wrapper.fit(X, y=y)

    def fit_transform(self, X, y=None, **fit_params):
        return self.data_frame_wrapper.fit_transform(X, y, **fit_params)

    def data_transform(self, X):
        return self.data_frame_wrapper.transform(X)


class StandardScaler(Transformer):
    def __init__(self, *, copy=True, with_mean=True, with_std=True):
        """creates a pyreal StandardScaler

        Args:
            copy (bool, optional): If False, try to avoid a copy and do inplace scaling instead.
                                   This is not guaranteed to always work inplace;
                                   e.g. if the data is not a NumPy array or scipy.sparse CSR matrix, a copy may still be returned.
            with_mean (bool, optional): If True, center the data before scaling.
                                        This does not work (and will raise an exception) when attempted on sparse matrices,
                                        because centering them entails building a dense matrix which in common use cases is likely to be too large to fit in memory.
            with_std (bool, optional): If True, scale the data to unit variance (or equivalently, unit standard deviation).
        """
        self.data_frame_wrapper = DataFrameWrapper(
            SklearnStandardScaler(copy=copy, with_mean=with_mean, with_std=with_std)
        )
        self.mean_ = None
        self.var_ = None

    def fit(self, X, y=None, sample_weight=None):
        ret = self.data_frame_wrapper.fit(X, y=y, sample_weight=sample_weight)
        self.mean_ = ret.wrapped_transformer.mean_
        self.var_ = ret.wrapped_transformer.var_
        return ret

    def data_transform(self, X):
        return self.data_frame_wrapper.transform(X)

    def inverse_transform(self, X):
        return self.data_frame_wrapper.inverse_transform(X)
