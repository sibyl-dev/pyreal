from abc import ABC

import numpy as np

from pyreal.explainers import ExplainerBase


class GlobalFeatureImportanceBase(ExplainerBase, ABC):
    """
    Base class for GlobalFeatureImportance explainer objects. Abstract class

    A GlobalFeatureImportanceBase object explains a machine learning prediction by assigning an
    importance score to every feature. GlobalFeatureImportanceBase objects explain
    by returning one number per feature, for the model in general.

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

    def __init__(self, model, x_train_orig=None, interpretable_features=True, **kwargs):
        self.interpretable_features = interpretable_features
        self.importance = None
        super(GlobalFeatureImportanceBase, self).__init__(model, x_train_orig, **kwargs)

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
                Not used for global Explainers

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
                    self.produce(self.x_train_orig_subset.iloc[0:n_rows]).get().to_numpy()
                )
        return np.max(np.var(explanations, axis=0))
