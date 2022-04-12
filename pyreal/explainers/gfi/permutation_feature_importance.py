import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from pyreal.explainers import GlobalFeatureImportanceBase
from pyreal.types.explanations.feature_based import FeatureImportanceExplanation


class PermutationFeatureImportance(GlobalFeatureImportanceBase):
    """
    PermutationFeatureImportance object.

    A PermutationFeatureImportance object gets a global feature explanation using the Permutation
    Feature Importance algorithm.

    Args:
        model (string filepath or model object):
           Filepath to the pickled model to explain, or model object with .predict() function
        x_train_orig (DataFrame of size (n_instances, n_features)):
            Training set in original form.
        **kwargs: see base Explainer args
    """

    def __init__(self, model, x_train_orig, **kwargs):
        self.explainer = None
        self.explainer_input_size = None
        super(PermutationFeatureImportance, self).__init__(model, x_train_orig, **kwargs)

    def fit(self):
        """
        Fit the feature importance explainer.
        No-op as permutation_importance does not require fitting
        """
        return self

    def get_importance(self):
        """
        Calculate the explanation of each feature using the permutation feature importance
        algorithm.

        Returns:
            DataFrame of shape (n_features, ):
                 The global importance of each feature
        """
        x = self.transform_to_x_model(self._x_train_orig)
        columns = x.columns
        x = np.asanyarray(x)
        importance_result = permutation_importance(self.model, x, self._y_orig, n_repeats=100)
        importances = importance_result.importances_mean
        return FeatureImportanceExplanation(
            pd.DataFrame(importances.reshape(1, -1), columns=columns)
        )
