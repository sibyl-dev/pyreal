from abc import ABC, abstractmethod

from pyreal.explainers import Explainer


class GlobalFeatureImportanceBase(Explainer, ABC):
    """
    Base class for GlobalFeatureImportance explainer objects. Abstract class

    A GlobalFeatureImportanceBase object explains a machine learning prediction by assigning an
    importance score to every feature. GlobalFeatureImportanceBase objects explain
    by returning one number per feature, for the model in general.

    Args:
        algorithm (ExplanationAlgorithm or None):
            Name of the algorithm this Explainer uses
        model (string filepath or model object):
           Filepath to the pickled model to explain, or model object with .predict() function
        x_orig (dataframe of shape (n_instances, x_orig_feature_count)):
           The training set for the explainer
        e_algorithm (string, one of ["shap"]):
           Explanation algorithm to use. If none, one will be chosen automatically based on model
           type
        interpretable_features (Boolean):
            If True, return explanations using the interpretable feature descriptions instead of
            default names
        **kwargs: see base Explainer args
    """

    def __init__(self, algorithm, model, x_orig, interpretable_features=True, **kwargs):
        self.interpretable_features = interpretable_features
        self.importance = None
        super(GlobalFeatureImportanceBase, self).__init__(algorithm, model, x_orig, **kwargs)

    @abstractmethod
    def fit(self):
        """
        Fit this explainer object
        """

    def produce(self, x_orig=None):
        """
        Produce the global feature importance explanation

        Args:
            x_orig (None):
                Global explanations do not take inputs - dummy to match signature

        Returns:
            DataFrame of shape (n_features,)
                Importance of each feature for the model
        """
        # Importance for a given model stays constant, so can be saved and re-returned
        if self.importance is not None:
            return self.importance
        importance = self.get_importance()
        importance = self.transform_explanation(importance)
        if self.interpretable_features:
            return self.convert_columns_to_interpretable(importance)
        self.importance = importance
        return importance

    @abstractmethod
    def get_importance(self):
        """
        Gets the raw explanation.

        Returns:
            DataFrame of shape (n_features, )
                Importance of each feature
        """
