from abc import ABC, abstractmethod

from pyreal.explainers import Explainer


class LocalFeatureContributionsBase(Explainer, ABC):
    """
    Base class for LocalFeatureContributionsBase explainer objects. Abstract class

    A LocalFeatureContributionsBase object explains a machine learning prediction by assigning an
    importance or contribution score to every feature. LocalFeatureContributionBase objects explain
    by taking an instance and returning one number per feature, per instance.

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
        super(LocalFeatureContributionsBase, self).__init__(algorithm, model, x_orig, **kwargs)

    @abstractmethod
    def fit(self):
        """
        Fit this explainer object
        """
        pass

    def produce(self, x_orig):
        """
        Produce the local feature contribution explanation

        Args:
            x_orig (DataFrame of shape (n_instances, n_features):
                Input to explain

        Returns:
            DataFrame of shape (n_instances, n_features)
                Contribution of each feature for each instance
        """
        if x_orig.ndim == 1:
            x_orig = x_orig.to_frame().T
        contributions = self.get_contributions(x_orig)
        contributions = self.transform_explanation(contributions)
        if self.interpretable_features:
            return self.convert_columns_to_interpretable(contributions)
        return contributions

    @abstractmethod
    def get_contributions(self, x_orig):
        """
        Gets the raw explanation explanation.
        Args:
            x_orig (DataFrame of shape (n_instances, n_features):
                Input to explain

        Returns:
            DataFrame of shape (n_instances, n_features)
                Contribution of each feature for each instance
        """
        pass
