import numpy as np
import pandas as pd

from pyreal.explainers.time_series import SaliencyBase
from pyreal.types.explanations.feature_based import FeatureContributionExplanation


class UnivariateOcclusionSaliency(SaliencyBase):
    """
    UnivariateOcclusionSaliency object.

    An OcclusionSaliency object judges the relative importance or saliency of each timestep
    value by iteratively occluding windows of data, and calculating the resulting change in model
    prediction.

    Can only take a single row input to .produce()
    """

    def __init__(
        self, model, x_train_orig, regression=False, width=5, k="avg", num_classes=None, **kwargs
    ):
        """
        Generates a feature importance explanation of time-series type data by iteratively
        occluding windows of data and computing the change in model prediction

        Args:
            model (string filepath or model object):
                Filepath to the pickled model to explain, or model object with .predict() function
            x_train_orig (DataFrame of size (n_instances, n_features)):
                Training set in original form.
            regression (Boolean):
                If true, model is a regression model.
                If false, must provide a num_classes or classes parameter
            width (int):
                Length of the occlusion window
            k (float or one of ["avg", "remove"]):
                The occlusion method. One of:
                    a float value - occlude with this constant value
                    "avg" - occlude with the average value of the window
                    "remove" - occlude by removing the section of the data
            num_classes (int):
                Required if regression=False and classes=None
            **kwargs: see base Explainer args
        """
        self.width = width
        self.k = k
        self.num_classes = num_classes
        if regression:
            self.num_classes = 1
        super(UnivariateOcclusionSaliency, self).__init__(model, x_train_orig, **kwargs)
        if self.num_classes is None and self.classes is None:
            raise ValueError(
                "Must provide classes or num_classes parameter when regression is False"
            )
        elif self.num_classes is None:
            self.num_classes = len(self.classes)

    def fit(self):
        return self

    def get_contributions(self, x_orig):
        """
        Calculate the explanation of each feature in x using occlusion.
        Args:
            x_orig (DataFrame of shape (1, n_features)):
               The input to be explained
        Returns:
            DataFrame of shape (n_classes, n_features):
                 The contribution of each feature to each class prediction.
        """
        x_algo = self.transform_to_x_algorithm(x_orig)
        if isinstance(x_algo, pd.Series):
            x_algo = x_algo.to_frame().T

        if x_algo.shape[0] > 1:
            raise ValueError(
                "UnivariateOcclusionSaliency.produce() can only take one row of input"
            )
        data_length = x_algo.shape[1]

        v = np.zeros((data_length, self.num_classes))
        sig = np.copy(x_algo)
        pred_orig = self.model.predict(self.transform_x_from_algorithm_to_model(sig)).reshape(-1)

        # Occlude the beginning of the sequence with smaller windows
        for i in range(1, self.width):
            pred = self._occlude_once(sig, 0, i, self.k)
            for j in range(0, i):
                v[j] += pred - pred_orig

        # Occlude the main body of the sequence with width-length windows
        for i in range(data_length - self.width + 1):
            pred = self._occlude_once(sig, i, i + self.width, self.k)
            for j in range(i, i + self.width):
                v[j] += pred - pred_orig

        # Occlude the end of the sequence with smaller windows
        for i in range(1, self.width):
            pred = self._occlude_once(sig, data_length - i, data_length, self.k)
            for j in range(data_length - i, data_length):
                v[j] += pred - pred_orig

        importance = v / self.width

        if isinstance(x_algo, pd.DataFrame):
            importances_df = pd.DataFrame(importance, index=x_algo.columns).T
        else:
            importances_df = pd.DataFrame(importance).T

        return FeatureContributionExplanation(importances_df)

    def _occlude_once(self, sig, win_min, win_max, k):
        """
        Occlude one window and return the resulting model prediction
        """
        occ_test_signal = np.copy(sig)
        if self.k == "avg":
            occ_test_signal[:, win_min:win_max] = np.average(
                [occ_test_signal[:, win_min], occ_test_signal[:, win_max - 1]]
            )
        elif self.k == "remove":
            mask = np.ones_like(occ_test_signal, dtype=bool)
            mask[:, win_min:win_max] = False
            new_shape = list(sig.shape)
            new_shape[1] = -1
            occ_test_signal = occ_test_signal[mask, ...].reshape(new_shape)
        else:
            occ_test_signal[:, win_min:win_max] = k

        pred = self.model.predict(
            self.transform_x_from_algorithm_to_model(occ_test_signal)
        ).reshape(-1)
        return pred
