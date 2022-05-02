import numpy as np
import pandas as pd
from shap import KernelExplainer, LinearExplainer, DeepExplainer

from pyreal.explainers import ClassificationSaliencyBase
from pyreal.types.explanations.feature_based import AdditiveFeatureContributionExplanation


def transform(X):
    X_pyreal = np.empty((X.shape[0], X.iloc[0][0].shape[0]))
    for i in range(X.shape[0]):
        for j in range(X.iloc[0][0].shape[0]):
            X_pyreal[i, j] = X.iloc[i][0][j]

    X_pyreal = pd.DataFrame(X_pyreal)
    return X_pyreal


class ShapImportance(ClassificationSaliencyBase):
    """
    IntervalImportance object.

    A IntervalImportance object creates features of time-series data by aggregating timestamps into
    intervals and produce explanation using the SHAP algorithm.

    IntervalImportance explainers expect data in the **model-ready feature space**

    Currently, only classification models explanation is supported.

    Args:
        model (string filepath or model object):
           Filepath to the pickled model to explain, or model object with .predict() function
        x_train_orig (DataFrame of size (n_instances, length of series)):
            Training set in original form.
        window_size (int):
            The size of the interval.
        shap_type (string, one of ["kernel", "linear"]):
            Type of shap algorithm to use. If None, SHAP will pick one.
        **kwargs: see base Explainer args
    """

    def __init__(self, model, x_train_orig,
                 window_size=1, shap_type=None, **kwargs):
        supported_types = ["kernel", "linear"]
        if shap_type is not None and shap_type not in supported_types:
            raise ValueError("Shap type not supported, given %s, expected one of %s or None" %
                             (shap_type, str(supported_types)))
        else:
            self.shap_type = shap_type

        self.window_size = window_size
        self.explainer = None
        self.explainer_input_size = None
        super(ShapImportance, self).__init__(model, x_train_orig, **kwargs)

    def fit(self):
        """
        Fit the contribution explainer
        """
        dataset = self.transform_to_x_algorithm(self._x_train_orig)
        self.explainer = KernelExplainer(self.model, dataset)


    def get_contributions(self, x_orig):
        """
        Calculate the explanation of each feature in x using SHAP.

        Args:
            x_orig (DataFrame of shape (n_instances, n_features)):
               The input to be explained
        Returns:
            DataFrame of shape (n_instances, n_features):
                 The contribution of each feature
        """
        sig = sig.reshape(1, 1, -1)  # .reshape(1, -1,1)

        shap_values = np.array(self.explainer.shap_values(sig))
        agg_shap_values = np.squeeze(np.mean(shap_values, axis=1))
        # agg_shap_values = (agg_shap_values - np.amin(agg_shap_values)) / (np.amax(agg_shap_values) - np.amin(agg_shap_values))

        return agg_shap_values[true_class].reshape(-1)
