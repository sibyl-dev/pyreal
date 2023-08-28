import numpy as np
import pandas as pd
from shap import Explainer as ShapExplainer
from shap import KernelExplainer, LinearExplainer, TreeExplainer

from pyreal.explainers import GlobalFeatureImportanceBase
from pyreal.explanation_types.explanations.feature_based import (
    AdditiveFeatureImportanceExplanation,
)


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

    def __init__(self, model, x_train_orig=None, shap_type=None, **kwargs):
        supported_types = ["kernel", "linear"]
        if shap_type is not None and shap_type not in supported_types:
            raise ValueError(
                "Shap type not supported, given %s, expected one of %s or None"
                % (shap_type, str(supported_types))
            )
        else:
            self.shap_type = shap_type

        self.explainer = None
        self.explainer_input_size = None
        self.importance_from_fit = None

        super(ShapFeatureImportance, self).__init__(model, x_train_orig, **kwargs)

    def fit(self, x_train_orig=None, y_train=None):
        """
        Fit the feature importance explainer

        Args:
            y_train:
            x_train_orig:
        """
        x_train_orig = self._get_x_train_orig(x_train_orig)

        x_model = self.transform_to_x_model(x_train_orig)
        self.explainer_input_size = x_model.shape[1]
        if self.shap_type == "kernel":
            self.explainer = KernelExplainer(self.model.predict, x_model)
        # Note: we manually check for linear model here because of SHAP bug
        elif self.shap_type == "linear":
            self.explainer = LinearExplainer(self.model, x_model)
        else:
            self.explainer = ShapExplainer(self.model, x_model)  # SHAP will pick an algorithm

        x_model_np = np.asanyarray(x_model)
        if isinstance(self.explainer, TreeExplainer):
            shap_values = np.array(self.explainer.shap_values(x_model_np, check_additivity=False))
        else:
            shap_values = np.array(self.explainer.shap_values(x_model_np))

        if shap_values.ndim < 2:
            raise RuntimeError("Something went wrong with SHAP - expected at least 2 dimensions")
        if shap_values.ndim > 2:
            predictions = self.model_predict(x_train_orig)

            if self.classes is not None:
                predictions = [np.where(self.classes == i)[0][0] for i in predictions]
            print(shap_values.shape)
            print(predictions)
            shap_values = shap_values[predictions, np.arange(shap_values.shape[1]), :]

        importances = np.mean(np.absolute(shap_values), axis=0).reshape(1, -1)
        self.importance_from_fit = pd.DataFrame(importances, columns=x_model.columns)

        return self

    def produce_explanation(self, **kwargs):
        """
        Calculate the explanation of each feature using SHAP.

        Returns:
            AdditiveFeatureImportanceExplanation:
                 The global importance of each feature
        """
        if self.importance_from_fit is None:
            raise RuntimeError("Must fit explainer before calling produce!")
        return AdditiveFeatureImportanceExplanation(self.importance_from_fit)
