import numpy as np
import pandas as pd

from pyreal.explainers import LocalFeatureContributionsBase
from pyreal.types.explanations.feature_based import FeatureContributionExplanation


class SimpleCounterfactualContribution(LocalFeatureContributionsBase):
    """
    SimpleCounterfactualContribution object.

    A SimpleCounterfactualContribution object gets feature contribution explanations by changing
    each feature to a set of other possible feature values through a random selection from the
    column, and then averaging the change in model prediction.

    Does not support classification models

    Expects categorical features rather than one-hot-encodings. Otherwise, can take any state.

    Args:
        model (string filepath or model object):
           Filepath to the pickled regression model to explain, or model object with .predict()
           function
        x_train_orig (DataFrame of size (n_instances, n_features)):
            Training set in original form.
        n_iterations (int):
            Number of samples to replace each feature with.
        **kwargs: see base Explainer args
    """

    def __init__(self, model, x_train_orig, n_iterations=30, **kwargs):
        self.explainer_input_size = None
        self.n_iterations = n_iterations
        super(SimpleCounterfactualContribution, self).__init__(model, x_train_orig, **kwargs)

    def fit(self):
        """
        Fit the contribution explainer
        """
        dataset = self.transform_to_x_algorithm(self._x_train_orig)
        self.explainer_input_size = dataset.shape[1]
        return self

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
        x = self.transform_to_x_algorithm(x_orig)
        if x.shape[1] != self.explainer_input_size:
            raise ValueError(
                "Received input of wrong size.Expected ({},), received {}".format(
                    self.explainer_input_size, x.shape
                )
            )
        x_train_explain = self.transform_to_x_algorithm(self._x_train_orig)
        pred_orig = self.model_predict_on_algorithm(x)
        contributions = pd.DataFrame(np.zeros_like(x), columns=x.columns)
        for col in x:
            total_abs_change = 0
            for i in range(self.n_iterations):
                x_copy = x.copy()
                x_copy[col] = x_train_explain[col].sample().iloc[0]
                pred_new = self.model_predict_on_algorithm(x_copy)
                total_abs_change += abs(pred_new - pred_orig)
            contributions[col] = total_abs_change / self.n_iterations
        return FeatureContributionExplanation(contributions)
