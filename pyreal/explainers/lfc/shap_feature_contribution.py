import numpy as np
import pandas as pd
from shap import Explainer as ShapExplainer
from shap import KernelExplainer, LinearExplainer

from pyreal.explainers import LocalFeatureContributionsBase
from pyreal.utils.transformer import ExplanationAlgorithm


class ShapFeatureContribution(LocalFeatureContributionsBase):
    """
    ShapFeatureContribution object.

    A ShapFeatureContribution object gets feature explanation using the SHAP algorithm.

    Args:
        model (string filepath or model object):
           Filepath to the pickled model to explain, or model object with .predict() function
        x_orig (DataFrame of size (n_instances, n_features)):
            Training set in original form.
        shap_type (string, one of ["kernel", "linear"]):
            Type of shap algorithm to use. If None, SHAP will pick one.
        **kwargs: see base Explainer args
    """
    def __init__(self, model, x_orig,
                 shap_type=None, **kwargs):
        supported_types = ["kernel", "linear"]
        if shap_type is not None and shap_type not in supported_types:
            raise ValueError("Shap type not supported, given %s, expected one of %s or None" %
                  (shap_type, str(supported_types)))
        else:
            self.shap_type = shap_type

        self.explainer = None
        self.algorithm = ExplanationAlgorithm.SHAP
        self.explainer_input_size = None
        super(ShapFeatureContribution, self).__init__(self.algorithm, model, x_orig, **kwargs)

    def fit(self):
        """
        Fit the contribution explainer
        """
        dataset = self.transform_to_x_explain(self.X_orig)
        self.explainer_input_size = dataset.shape[1]
        if self.shap_type == "kernel":
            self.explainer = KernelExplainer(self.model.predict, dataset)
        # Note: we manually check for linear model here because of SHAP bug
        elif self.shap_type == "linear":
            self.explainer = LinearExplainer(self.model, dataset)
        else:
            self.explainer = ShapExplainer(self.model, dataset)  # SHAP will pick an algorithm

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
        if self.explainer is None:
            raise AttributeError("Instance has no explainer. Must call "
                                 "fit_contribution_explainer before "
                                 "get_contributions")
        x = self.transform_to_x_explain(x_orig)
        if x.shape[1] != self.explainer_input_size:
            raise ValueError("Received input of wrong size."
                             "Expected ({},), received {}"
                             .format(self.explainer_input_size, x.shape))
        columns = x.columns
        x = np.asanyarray(x)
        return pd.DataFrame(self.explainer.shap_values(x), columns=columns)
