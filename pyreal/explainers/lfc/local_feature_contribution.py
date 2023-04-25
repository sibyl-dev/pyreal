import logging

from pyreal.explainers import (
    LocalFeatureContributionsBase,
    ShapFeatureContribution,
    SimpleCounterfactualContribution,
)

log = logging.getLogger(__name__)


def choose_algorithm():
    """
    Choose an algorithm based on the model type.
    Currently, shap is the only supported algorithm

    Returns:
        string (one of ["shap"])
            Explanation algorithm to use
    """
    return "shap"


class LocalFeatureContribution(LocalFeatureContributionsBase):
    """
    Generic LocalFeatureContribution wrapper

    A LocalFeatureContributions object wraps multiple local feature-based explanations. If no
    specific algorithm is requested, one will be chosen based on the information given.
    Currently, only SHAP is supported.

    Args:
        model (string filepath or model object):
           Filepath to the pickled model to explain, or model object with .predict() function
        x_train_orig (dataframe of shape (n_instances, x_orig_feature_count)):
           The training set for the explainer
        e_algorithm (string, one of ["shap", "simple"]):
           Explanation algorithm to use. If none, one will be chosen automatically based on model
           type
        shap_type (string, one of ["kernel", "linear"]):
            Type of shap algorithm to use, if e_algorithm="shap".
        **kwargs: see LocalFeatureContributionsBase args
    """

    def __init__(self, model, x_train_orig=None, e_algorithm=None, shap_type=None, **kwargs):
        if e_algorithm is None:
            e_algorithm = choose_algorithm()
        self.base_local_feature_contribution = None
        if e_algorithm == "shap":
            self.base_local_feature_contribution = ShapFeatureContribution(
                model, x_train_orig, shap_type=shap_type, **kwargs
            )
        elif e_algorithm == "simple":
            self.base_local_feature_contribution = SimpleCounterfactualContribution(
                model, x_train_orig, **kwargs
            )
        if self.base_local_feature_contribution is None:
            raise ValueError("Invalid algorithm type %s" % e_algorithm)

        super(LocalFeatureContribution, self).__init__(model, x_train_orig, **kwargs)

    def fit(self, x_train_orig=None, y_train=None):
        """
        Fit this explainer object

        Args:
            x_train_orig (DataFrame of shape (n_instances, n_features):
                Training set to fit on, required if not provided on initialization
            y_train:
                Targets of training set, required if not provided on initialization
        """
        self.base_local_feature_contribution.fit(x_train_orig, y_train)
        return self

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
        return self.base_local_feature_contribution.get_contributions(x_orig)
