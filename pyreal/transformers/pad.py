import numpy as np
import pandas as pd

from pyreal.transformers import Transformer


class TimeSeriesPadder(Transformer):
    """
    A transformer that pads and truncates variable-length time series to equal lengths
    """

    def __init__(self, value, length=None, **kwargs):
        """
        Initializes the transformer

        Args:
            length (int):
                Length to pad/truncate time series sequences to. If none, pad to maximum length in
                fitting dataset
            value (Union[int, float, complex, ndarray, Iterable]):
                Object value to pad with

        """
        if length is not None and length <= 0:
            raise ValueError("Length must be integer >= 0")
        self.length = length
        self.value = value
        super().__init__(**kwargs)

    def fit(self, x, **params):
        """
        Determines the length to pad to if not set

        Args:
            x (DataFrame or numpy array of shape (n_instances, n_features)):
                The dataset to fit on
        Returns:
            None

        """
        if self.length is None:
            if isinstance(x, pd.DataFrame):
                self.length = x.shape[1]
            else:
                self.length = len(max(x, key=lambda x_: len(x_)))
        super().fit(x)

    def data_transform(self, x):
        """
        Reorders and selects the features in x. If no length has been set
        and the transformer has not been fit, pad to the longest subsequence length

        Args:
            x (numpy array of shape (n_instances, n_features)):
                The data to transform
        Returns:
            numpy array of shape (n_instances, len(columns)):
                The data with features selected and reordered
        """
        if self.length is None:
            length = len(max(x, key=lambda x_: len(x_)))
        else:
            length = self.length
        z = np.full([len(x), length], self.value)
        for i, j in enumerate(x):
            if len(j) < z.shape[1]:
                z[i][0 : len(j)] = j
            else:
                z[i][0:length] = j[0:length]
        return z
