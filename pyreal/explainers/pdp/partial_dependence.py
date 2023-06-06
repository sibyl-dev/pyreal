from sklearn.inspection import partial_dependence

from pyreal.explainers import PartialDependenceExplainerBase
from pyreal.types.explanations.feature_value_based import PartialDependenceExplanation


class PartialDependence(PartialDependenceExplainerBase):
    """
    PartialDependence Object

    A PartialDependence object explains a machine learning prediction by showing the
    marginal effect each feature has on the model prediction.

    Args:
        model (string filepath or model object):
           Filepath to the pickled model to explain, or model object with .predict() function
        x_train_orig (dataframe of shape (n_instances, x_orig_feature_count)):
           The training set for the explainer
        features (list of string(s)):
            The features to be explained.
        interpretable_features (Boolean):
            If True, return explanations using the interpretable feature descriptions instead of
            default names
        **kwargs: see base Explainer args
    """

    def __init__(
        self,
        model,
        features,
        x_train_orig=None,
        grid_resolution=100,
        interpretable_features=True,
        **kwargs,
    ):
        self.features = features
        self.grid_resolution = grid_resolution
        super().__init__(model, x_train_orig, interpretable_features, **kwargs)

    def fit(self, x_train_orig=None, y_train=None):
        """
        Fit this explainer.

        Calculates the partial dependence values

        Values:
            self.pdp_values: ndarray of shape (n_instances, grid_points[0], grid_points[1],...)
                Partial dependence values calculated for each feature combination in the grid.
            self.grid_points: ndarray of shape (features, grid_resolution)
                The grid points where the partial dependence values are calculated.

        Args:
            x_train_orig (DataFrame of shape (n_instances, n_features):
                Training set to fit on, required if not provided on initialization
            y_train:
                Targets of training set, required if not provided on initialization
        """
        x_train_orig = self._get_x_train_orig(x_train_orig)

        dataset = self.transform_to_x_model(x_train_orig)
        explanation_results = partial_dependence(
            self.model,
            dataset,
            features=self.features,
            grid_resolution=self.grid_resolution,
        )
        self.pdp_values = explanation_results["average"]
        self.grid_points = explanation_results["values"]
        return self

    def get_pdp(self):
        """
        Produce the partial dependence explanation

        """
        return PartialDependenceExplanation(self.features, self.pdp_values, self.grid_points)
