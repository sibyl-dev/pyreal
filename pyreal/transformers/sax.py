# Code adapted form sktime.
# source:
# https://github.com/alan-turing-institute/sktime/tree/v0.9.0/sktime/transformations/panel/dictionary_based
import numpy as np
import pandas as pd

from pyreal.transformers import Transformer
from pyreal.transformers.time_series_formatter import MultiIndexFrameToNumpy3d, is_valid_dataframe


class SAXTransformer(Transformer):
    """
    Symbolic Aggregate approXimation.

    Args:
        n_bins (Integer, default = 4):
            The number of bins to produce. It must be between 2 and
            ``min(n_timestamps, 26)``.
        window_size (Integer):
            Size of window slided across the signal.
        word_length (Integer):
            The number of characters in each representation of word, which
            is reduced from each window.

    Reference:
    [1] J. Lin, E. Keogh, L. Wei, and S. Lonardi, "Experiencing SAX: a
        novel symbolic representation of time series". Data Mining and
        Knowledge Discovery, 15(2), 107-144 (2007).
    [2] Yifeng Gao and Jessica Lin. Efficient discovery of variable-length
        time series motifs with large length range in million scale time
        series. arXiv preprint arXiv:1802.04883, 2018
    """

    def __init__(self, n_bins=4, window_size=32, word_length=8, **kwargs):
        self.n_bins = n_bins
        self.window_size = window_size
        self.word_length = word_length

        super().__init__(**kwargs)

    def data_transform(self, x):
        """
        Bin the data with the given alphabet.

        Args:
            x (DataFrame of shape (n_instances, n_timestamp) or MultiIndex-column DataFrame):
                Univariate time series or Multivariate time series data.

        Returns:
            X_new : array, shape = (n_samples, n_timestamps)
                Binned data.
        """
        if not isinstance(x, pd.DataFrame) and not isinstance(x, np.ndarray):
            raise TypeError(
                f"Input data must be a pd.DataFrame or np.ndarray, but found {type(x)}"
            )

        if is_valid_dataframe(x):
            n_timestamps = x.columns.levshape[1]
            data = MultiIndexFrameToNumpy3d().fit_transform(x)
        elif isinstance(x, pd.DataFrame):
            n_timestamps = x.shape[1]
            data = x.to_numpy()[:, np.newaxis, :]

        paaList = self.paa_transform(data, n_timestamps)
        saxList = self.paa_to_sax(paaList)
        # TODO: format SAX into pandas dataframe?
        return saxList

    def intervals(self):
        # Pre-made gaussian curve breakpoints from UEA TSC codebase
        interval_map = {
            2: [0],
            3: [-0.43, 0.43],
            4: [-0.67, 0, 0.67],
            5: [-0.84, -0.25, 0.25, 0.84],
            6: [-0.97, -0.43, 0, 0.43, 0.97],
            7: [-1.07, -0.57, -0.18, 0.18, 0.57, 1.07],
            8: [-1.15, -0.67, -0.32, 0, 0.32, 0.67, 1.15],
            9: [-1.22, -0.76, -0.43, -0.14, 0.14, 0.43, 0.76, 1.22],
            10: [
                -1.28,
                -0.84,
                -0.52,
                -0.25,
                0.0,
                0.25,
                0.52,
                0.84,
                1.28,
            ],
        }
        return interval_map[self.n_bins]

    def paa_transform(self, x, n_timestamps, std_thresh=1e-3, save_stats=True):
        """
        Perform Piecewise Aggregate Approximation.

        Args:
            x (ndarray of shape (n_instances, n_variables, n_timestamps)):
                Univariate time series or Multivariate time series data.
            std_thresh (float):
                Small number to stop values from blowing up after normalization.
        Returns:
            ndarray of shape (n_instances, n_variables, n_sequences, word_length)
                PAA representation of series.
        """
        if len(x.shape) != 3:
            raise ValueError(
                f"The number of dimensions of x should be 3, but x has shape {x.shape}."
            )
        # The last axis is the timestamp axis
        self.cum_sum = np.concatenate(
            (np.zeros(x[..., 0:1].shape), np.cumsum(x, axis=-1)), axis=-1
        )
        self.cum_square_sum = np.concatenate(
            (np.zeros(x[..., 0:1].shape), np.cumsum(x**2, axis=-1)), axis=-1
        )

        # calculate breakpoints for PAA segments
        step = self.window_size // self.word_length
        res = self.window_size % self.word_length
        break_indices = [*range(0, self.window_size, step)]
        # handle divisibility
        if res == 0:
            break_indices.append(self.window_size)
            break_indices = np.array(break_indices)
        else:
            break_indices = np.array(break_indices)
            break_indices[-res:] += 1 + np.arange(res)

        #
        n_sequences = n_timestamps - self.window_size + 1

        # seqSum and seqSquareSum have shape (n_instances, n_variables, n_sequences)
        seqSum = self.cum_sum[..., self.window_size :] - self.cum_sum[..., :n_sequences]
        seqSquareSum = (
            self.cum_square_sum[..., self.window_size :] - self.cum_square_sum[..., :n_sequences]
        )
        seqMean = seqSum / self.window_size
        seqStd = np.sqrt(seqSquareSum / self.window_size - seqMean**2)

        # paa_sum and paa has shape (n_instances, n_variables, n_sequences, word_length)
        break_indices_matrix = np.arange(n_sequences)[:, None] + break_indices[None, :]
        paa_sum = (
            self.cum_sum[..., break_indices_matrix[:, 1:]]
            - self.cum_sum[..., break_indices_matrix[:, :-1]]
        )

        paa = (paa_sum * self.word_length / self.window_size - seqMean[:, :, :, None]) / (
            seqStd[:, :, :, None] + std_thresh
        )

        if save_stats:
            self.seqMean = seqMean
            self.seqStd = seqStd

        return paa

    def paa_to_sax(self, paa):
        """
        Convert PAA representation to SAX.

        Args:
            paa (ndarray of shape (n_instances, n_variables, n_sequences, word_length))
                PAA representation.
        Returns:
            ndarray of shape (n_instances, n_variables, n_sequences, word_length)
                SAX representation, in the form of integers (e.g. 01232012)

        """
        split = self.intervals()
        sax = np.zeros_like(paa)
        for endpoint in split:
            sax[paa > endpoint] += 1
        return sax.astype(int)

    def get_original_mean(self):
        return self.seqMean

    def get_original_std(self):
        return self.seqStd
