import numpy as np
import pandas as pd
from shap import KernelExplainer, LinearExplainer

from pyreal.explainers import TimeSeriesImportanceBase
from pyreal.types.explanations.feature_based import AdditiveFeatureContributionExplanation


class OcclusionImportance(TimeSeriesImportanceBase):
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
                 window_size=1, **kwargs):
        self.window_size = window_size
        self.explainer_input_size = None
        super(OcclusionImportance, self).__init__(model, x_train_orig, **kwargs)

    def fit(self):
        """
        Fit the contribution explainer
        """
        return self

    def get_contributions(self, x_orig, width=5, k="avg"):
        """
        Calculate the explanation of each feature in x using SHAP.

        Args:
            x_orig (DataFrame of shape (n_instances, n_features)):
               The input to be explained
        Returns:
            DataFrame of shape (n_instances, n_features):
                 The contribution of each feature
        """
        data_length = x_orig.shape[1]
        print(data_length)
        v = np.zeros((x_orig.shape[1], len(self.classes)))
        sig = x_orig

        for i in range(1, width):
            occ_test_signal = np.copy(sig)
            if(k == "avg"):
                occ_test_signal[:, 0:i] = np.average([occ_test_signal[:,0], occ_test_signal[:,i-1]])
            else:
                occ_test_signal[:, 0:i] = k
            pred = self.model.predict_proba(occ_test_signal).reshape(-1)
            #for j in range(0, i):
            for j in range(i, data_length):
                v[j] += pred

        for i in range(data_length - width):
            occ_test_signal = np.copy(sig)
            if (k == "avg"):
                occ_test_signal[:, i:i + width] = np.average([occ_test_signal[:, i], occ_test_signal[:, i+width-1]])
            else:
                occ_test_signal[:, i:i + width] = k
            pred = self.model.predict_proba(occ_test_signal).reshape(-1)
            #for j in range(i, i + width):
            for j in range(0, i):
                v[j] += pred
            for j in range(i+width, data_length):
                v[j] += pred

        for i in range(1, width):
            occ_test_signal = np.copy(sig)
            if (k == "avg"):
                occ_test_signal[:, data_length - i:data_length] = \
                    np.average([occ_test_signal[:, data_length-i], occ_test_signal[:, data_length-1]])
            else:
                occ_test_signal[:, data_length - i:data_length] = k
            pred = self.model.predict_proba(occ_test_signal).reshape(-1)
            #for j in range(data_length - i, data_length):
            for j in range(0, data_length - i):
                v[j] += pred

        print(v)

        predicted_class = self.model.predict(sig)[0][0]
        class_index = self.classes.index(predicted_class)
        importance = v[:, class_index]
        print(importance)
        #max = width
        #min = 0
        max = np.amax(importance)
        min = np.amin(importance)
        importance = (importance - min) / (max - min)
        #importance = 1 - importance

        return importance.reshape(-1)
