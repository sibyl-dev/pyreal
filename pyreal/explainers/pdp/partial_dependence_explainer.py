from pyreal.explainers import PartialDependence, PartialDependenceExplainerBase


class PartialDependenceExplainer(PartialDependenceExplainerBase):
    """
    Generic PartialDependence wrapper

    A PartialDependenceExplainer object explains a machine learning prediction by showing the
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

    def __init__(self, model, x_train_orig, features, grid_resolution=100, **kwargs):
        self.base_partial_dependence = PartialDependence(
            model, x_train_orig, features=features, grid_resolution=grid_resolution
        )
        super(PartialDependenceExplainer, self).__init__(model, x_train_orig, **kwargs)

    def get_pdp(self):
        """
        Gets the raw explanation

        Returns:
            PDP explanation object.
        """
        return self.base_partial_dependence.get_pdp()

    def fit(self):
        """
        Fit this explainer object
        """
        self.base_partial_dependence.fit()
        return self
