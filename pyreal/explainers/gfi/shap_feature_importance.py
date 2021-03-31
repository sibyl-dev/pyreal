import numpy as np
import pandas as pd
from shap import Explainer as ShapExplainer
from shap import KernelExplainer, LinearExplainer

from pyreal.explainers import GlobalFeatureImportanceBase
from pyreal.utils.transformer import ExplanationAlgorithm


class ShapFeatureImportance(GlobalFeatureImportanceBase):
    """
    ShapFeatureImportance object.

    A ShapFeatureImportance object gets feature explanation using the SHAP algorithm.

    Args:
        model (string filepath or model object):
           Filepath to the pickled model to explain, or model object with .predict() function
        x_train_orig (DataFrame of size (n_instances, n_features)):
            Training set in original form.
        shap_type (string, one of ["kernel", "linear"]):
            Type of shap algorithm to use. If None, SHAP will pick one.
        **kwargs: see base Explainer args
    """

    def __init__(self, model, x_train_orig,
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
        super(ShapFeatureImportance, self).__init__(self.algorithm, model, x_train_orig, **kwargs)

    def fit(self):
        """
        Fit the contribution explainer
        """
        dataset = self.transform_to_x_explain(self.x_train_orig)
        self.explainer_input_size = dataset.shape[1]
        if self.shap_type == "kernel":
            self.explainer = KernelExplainer(self.model.predict, dataset)
        # Note: we manually check for linear model here because of SHAP bug
        elif self.shap_type == "linear":
            self.explainer = LinearExplainer(self.model, dataset)
        else:
            self.explainer = ShapExplainer(self.model, dataset)  # SHAP will pick an algorithm

    def get_importance(self):
        """
        Calculate the explanation of each feature using SHAP.

        Returns:
            DataFrame of shape (n_features, ):
                 The global importance of each feature
        """
        if self.explainer is None:
            raise AttributeError("Instance has no explainer. Must call "
                                 "fit_contribution_explainer before "
                                 "get_contributions")
        x = self.transform_to_x_explain(self.x_train_orig)
        columns = x.columns
        x = np.asanyarray(x)
        all_contributions = self.explainer.shap_values(x)
        importances = np.mean(np.absolute(all_contributions), axis=0).reshape(1, -1)
        return pd.DataFrame(importances, columns=columns)