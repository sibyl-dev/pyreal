from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from pyreal.explainers import ExplainerBase


class SaliencyBase(ExplainerBase, ABC):
    """
    Base class for time series saliency explainer objects. Abstract class

    A SaliencyBase object explains a time-series ML classification or regression model

    Args:
        model (string filepath or model object):
           Filepath to the pickled model to explain, or model object with .predict() function
           .predict() should return probabilities of classes for classification,
            or numeric outputs for regression
        x_train_orig (dataframe of shape (n_instances, length of series)):
           The training set for the explainer
        **kwargs: see base Explainer args
    """

    def __init__(self, model, x_train_orig=None, interpretable_features=True, **kwargs):
        self.interpretable_features = interpretable_features
        super(SaliencyBase, self).__init__(model, x_train_orig, **kwargs)

    @abstractmethod
    def fit(self, x_train_orig=None, y_train=None):
        """
        Fit this explainer object

        Args:
            x_train_orig (DataFrame of shape (n_instances, n_features):
                Training set to fit on, required if not provided on initialization
            y_train:
                Targets of training set, required if not provided on initialization
        """

    def evaluate_variation(self, with_fit=False, explanations=None, n_iterations=20, n_rows=10):
        """
        Evaluate the variation of the explanations generated by this Explainer.
        A variation of 0 means this explainer is expected to generate the exact same explanation
        given the same model and input. Variation is always non-negative, and can be arbitrarily
        high.

        Args:
            with_fit (Boolean):
                If True, evaluate the variation in explanations including the fit (fit each time
                before running). If False, evaluate the variation in explanations of a pre-fit
                Explainer.
            explanations (None or List of DataFrames of shape (n_instances, n_features)):
                If provided, run the variation check on the precomputed list of explanations
                instead of generating
            n_iterations (int):
                Number of explanations to generate to evaluation variation
            n_rows (int):
                Number of rows of dataset to generate explanations on

        Returns:
            float
                The variation of this Explainer's explanations
        """
        if explanations is None:
            explanations = []
            for _ in range(n_iterations - 1):
                if with_fit:
                    self.fit()
                explanations.append(self.produce(self._x_train_orig.iloc[0:n_rows])[0].to_numpy())

        return np.max(np.var(explanations, axis=0))
