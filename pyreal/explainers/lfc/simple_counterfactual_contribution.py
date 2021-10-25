import numpy as np
import pandas as pd

from pyreal.explainers import LocalFeatureContributionsBase
from pyreal.utils.explanation_algorithm import ExplanationAlgorithm


class SimpleCounterfactualContribution(LocalFeatureContributionsBase):
    """
    SimpleCounterfactualContribution object.

    A SimpleCounterfactualContribution object gets feature contribution explanations by changing
    each feature to a set of other possible feature values through a random selection from the
    column, and then averaging the change in model prediction.

    Args:
        model (string filepath or model object):
           Filepath to the pickled model to explain, or model object with .predict() function
        x_train_orig (DataFrame of size (n_instances, n_features)):
            Training set in original form.
        shap_type (string, one of ["kernel", "linear"]):
            Type of shap algorithm to use. If None, SHAP will pick one.
        **kwargs: see base Explainer args
    """

    def __init__(self, model, x_train_orig, **kwargs):
        self.algorithm = ExplanationAlgorithm.PERMUTATION_IMPORTANCE
        self.explainer_input_size = None
        super(SimpleCounterfactualContribution, self).__init__(
            self.algorithm, model, x_train_orig, **kwargs)

    def fit(self):
        """
        Fit the contribution explainer
        """
        dataset = self.transform_to_x_explain(self.x_train_orig)
        self.explainer_input_size = dataset.shape[1]

    def get_contributions(self, x_orig, n=30):
        """
        Calculate the explanation of each feature in x using SHAP.

        Args:
            x_orig (DataFrame of shape (n_instances, n_features)):
               The input to be explained
            n (int)
        Returns:
            DataFrame of shape (n_instances, n_features):
                 The contribution of each feature
        """
        x = self.transform_to_x_explain(x_orig)
        if x.shape[1] != self.explainer_input_size:
            raise ValueError("Received input of wrong size."
                             "Expected ({},), received {}"
                             .format(self.explainer_input_size, x.shape))
        x_train_explain = self.transform_to_x_explain(self.x_train_orig)
        pred_orig = self.model_predict(x)
        contributions = pd.DataFrame(np.zeros_like(x), columns=x.columns)
        for col in x:
            total_abs_change = 0
            for i in range(n):
                x_copy = x.copy()
                x_copy[col] = x_train_explain[col].sample().iloc[0]
                pred_new = self.model_predict(x_copy)
                total_abs_change += abs(pred_new - pred_orig)
            contributions[col] = total_abs_change / n
        return contributions

