from abc import ABC, abstractmethod

import numpy as np

from pyreal.explainers import ExplainerBase


class LocalFeatureContributionsBase(ExplainerBase, ABC):
    """
    Base class for LocalFeatureContributions explainer objects. Abstract class

    A LocalFeatureContributionsBase object explains a machine learning prediction by assigning an
    importance or contribution score to every feature. LocalFeatureContributionBase objects explain
    by taking an instance and returning one number per feature, per instance.

    Args:
        model (string filepath or model object):
           Filepath to the pickled model to explain, or model object with .predict() function
        x_train_orig (dataframe of shape (n_instances, x_orig_feature_count)):
           The training set for the explainer
        interpretable_features (Boolean):
            If True, return explanations using the interpretable feature descriptions instead of
            default names
        **kwargs: see base Explainer args
    """

    def __init__(self, model, x_train_orig, interpretable_features=True, **kwargs):
        self.interpretable_features = interpretable_features
        super(LocalFeatureContributionsBase, self).__init__(model, x_train_orig, **kwargs)

    @abstractmethod
    def fit(self):
        """
        Fit this explainer object
        """

    def produce(self, x_orig):
        """
        Produce the local feature contribution explanation

        Args:
            x_orig (DataFrame of shape (n_instances, n_features)):
                Input to explain

        Returns:
            DataFrame of shape (n_instances, n_features)
                Contribution of each feature for each instance
        """
        if x_orig.ndim == 1:
            x_orig = x_orig.to_frame().T
        contributions = self.get_contributions(x_orig)
        contributions = self.transform_explanation(contributions).get()
        if self.interpretable_features:
            return self.convert_columns_to_interpretable(contributions)
        return contributions

    @abstractmethod
    def get_contributions(self, x_orig):
        """
        Gets the raw explanation.
        Args:
            x_orig (DataFrame of shape (n_instances, n_features):
                Input to explain

        Returns:
            DataFrame of shape (n_instances, n_features)
                Contribution of each feature for each instance
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
            for i in range(n_iterations - 1):
                if with_fit:
                    self.fit()
                explanations.append(
                    self.produce(self.x_train_orig.iloc[0:n_rows]).to_numpy())
        return np.max(np.var(explanations, axis=0))
