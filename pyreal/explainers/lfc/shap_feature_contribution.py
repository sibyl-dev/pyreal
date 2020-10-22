import numpy as np
import pandas as pd
from shap import Explainer as ShapExplainer
from shap import KernelExplainer
from shap import LinearExplainer

from pyreal.explainers import LocalFeatureContributionsBase


class ShapFeatureContribution(LocalFeatureContributionsBase):
    """
    ShapFeatureContribution object.

    A ShapFeatureContribution object gets feature contributions using the SHAP algorithm.

    Args:
        model_pickle_filepath (string filepath):
            Filepath to the pickled model to explain
        x_orig (DataFrame of size (n_instances, n_features)):
            Training set in original form.
        contribution_transforms (transformer or list of transformers):
            Transformer that convert contributions from explanation form to interpretable form
        shap_type (string, one of ["kernel", "linear"]):
            Type of shap algorithm to use. If None, SHAP will pick one.
        **kwargs: see base Explainer args
    """
    def __init__(self, model_pickle_filepath, x_orig,
                 shap_type=None, **kwargs):
        supported_types = ["kernel", "linear"]
        if shap_type is not None and shap_type not in supported_types:
            raise ValueError("Shap type not supported, given %s, expected one of %s or None" %
                  (shap_type, str(supported_types)))
        else:
            self.shap_type = shap_type

        self.explainer = None
        super(ShapFeatureContribution, self).__init__(model_pickle_filepath, x_orig, **kwargs)

    def fit(self):
        """
        Fit the contribution explainer
        """
        dataset = self.transform_to_x_explain(self.X_orig)
        if self.shap_type == "kernel":
            self.explainer = KernelExplainer(self.model.predict, dataset)
        # Note: we manually check for linear model here because of SHAP bug
        elif self.shap_type == "linear" or LinearExplainer.supports_model(self.model):
            self.explainer = LinearExplainer(self.model, dataset)
        else:
            self.explainer = ShapExplainer(self.model, dataset)  # SHAP will pick an algorithm

    def get_contributions(self, x_orig):
        """
        Calculate the contributions of each feature in x using SHAP.

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
        columns = x.columns
        x = np.asanyarray(x)
        return pd.DataFrame(self.explainer.shap_values(x), columns=columns)
