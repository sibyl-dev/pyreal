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

    def __init__(self, model, features, x_train_orig=None, grid_resolution=100, **kwargs):
        self.base_partial_dependence = PartialDependence(
            model, features=features, x_train_orig=x_train_orig, grid_resolution=grid_resolution
        )
        super(PartialDependenceExplainer, self).__init__(model, x_train_orig, **kwargs)

    def get_pdp(self):
        """
        Gets the raw explanation

        Returns:
            PDP explanation object.
        """
        return self.base_partial_dependence.get_pdp()

    def fit(self, x_train_orig=None, y_train=None):
        """
        Fit this explainer object

        Args:
            x_train_orig (DataFrame of shape (n_instances, n_features):
                Training set to fit on, required if not provided on initialization
            y_train:
                Targets of training set, required if not provided on initialization
        """
        self.base_partial_dependence.fit(x_train_orig, y_train)
        return self
