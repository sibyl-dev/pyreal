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

    def __init__(self, model, x_train_orig=None, **kwargs):
        self.explainer = None
        self.explainer_input_size = None
        self.importance_from_fit = None

        super(PermutationFeatureImportance, self).__init__(model, x_train_orig, **kwargs)

    def fit(self, x_train_orig=None, y_train=None):
        """
        Fit the feature importance explainer.
        No-op as permutation_importance does not require fitting

        Args:
            y_train:
            x_train_orig:
        """
        x_train_orig, y_train = self._get_training_data(x_train_orig, y_train)

        x = self.transform_to_x_model(x_train_orig)
        columns = x.columns
        x = np.asanyarray(x)
        importance_result = permutation_importance(self.model, x, y_train, n_repeats=100)
        importances = importance_result.importances_mean
        self.importance_from_fit = pd.DataFrame(importances.reshape(1, -1), columns=columns)

        return self

    def get_importance(self):
        """
        Calculate the explanation of each feature using the permutation feature importance
        algorithm.

        Returns:
            DataFrame of shape (n_features, ):
                 The global importance of each feature
        """
        if self.importance_from_fit is None:
            raise RuntimeError("Must fit explainer before calling produce!")
        return FeatureImportanceExplanation(self.importance_from_fit)
