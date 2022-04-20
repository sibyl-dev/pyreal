import numpy as np
import pandas as pd
from shap import KernelExplainer, LinearExplainer, DeepExplainer

from pyreal.explainers import TimeSeriesImportanceBase
from pyreal.types.explanations.feature_based import AdditiveFeatureContributionExplanation

import keras.backend as K
import tensorflow as tf

from pyreal.types.explanations.time_series_saliency import TimeSeriesSaliency


def transform(X):
    X_pyreal = np.empty((X.shape[0], X.iloc[0][0].shape[0]))
    for i in range(X.shape[0]):
        for j in range(X.iloc[0][0].shape[0]):
            X_pyreal[i, j] = X.iloc[i][0][j]

    X_pyreal = pd.DataFrame(X_pyreal)
    return X_pyreal


class MaskImportance(TimeSeriesImportanceBase):
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
        super(MaskImportance, self).__init__(model, x_train_orig, **kwargs)

    def fit(self):
        """
        Fit the contribution explainer
        """
        return self

    def get_contributions(self, x_orig, max_iterations=10, k=0.01, l1_coeff=0.01, l2_coeff=0.001):
        """


        Args:
            x_orig (DataFrame of shape (n_instances, n_features)):
               The input to be explained
        Returns:
            DataFrame of shape (n_instances, n_features):
                 The contribution of each feature
        """
        predicted_class = self.model_predict_on_algorithm(x_orig)
        sig = K.constant(x_orig)

        mask_init = np.ones(sig.shape, dtype=np.float32)

        #mask_init = config.m_reshape(np.ones((sig.shape[1]), dtype=np.float32))#.reshape(1, -1, 1)
        mask = K.variable(mask_init)

        for j in range(max_iterations):
            print(str(j) + "/" + str(max_iterations))

            perturbated_input = (sig * mask) + (k * (1 - mask))
            outputs = self.model_predict_on_algorithm(perturbated_input.numpy())

            loss = l1_coeff * K.mean(K.abs(1 - mask)) + \
                   l2_coeff * K.sum(mask[1:] - mask[:-1]) + \
                   outputs[:, predicted_class]

            grads = K.gradients(loss, mask)[0]
            grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
            mask = mask - (grads)

            # loss = K.stop_gradient(loss)

            mask = tf.clip_by_value(mask, 0, 1)

        sig = K.eval(sig).reshape(-1)
        importance = K.eval(1 - mask).reshape(-1)
        max = np.amax(importance)
        min = np.amin(importance)
        importance = (importance - min) / (max - min)

        return TimeSeriesSaliency(importance)
