import logging

from pyreal.explainers import PartialDependence, PartialDependenceExplainerBase

log = logging.getLogger(__name__)


def pdp(
    return_explanation=True,
    return_explainer=False,
    explainer=None,
    model=None,
    x_train_orig=None,
    features=None,
    feature_descriptions=None,
    e_transforms=None,
    m_transforms=None,
    i_transforms=None,
    interpretable_features=True,
):
    """
    Get partial dependence for input data (x_orig)

    Args:
        return_explanation (Boolean):
            If true, return explanation of features in x_input.
            If true, requires one of `explainer` or (`model and x_train`) and `features`
        return_explainer (Boolean):
            If true, return the fitted Explainer object.
            If true, requires one of `explainer` or (`model and x_train`)
        explainer (Explainer):
            Fitted explainer object.
        model (string filepath or model object):
           Filepath to the pickled model to explain, or model object with .predict() function
        x_train_orig (dataframe of shape (n_instances, x_orig_feature_count)):
           The training set for the explainer
        features (list of strings):
           features to be explained
        feature_descriptions (dict):
           Interpretable descriptions of each feature
        e_transforms (transformer object or list of transformer objects):
           Transformer(s) that need to be used on x_orig for the explanation algorithm:
           x_orig -> x_algorithm
        m_transforms (transformer object or list of transformer objects):
           Transformer(s) needed on x_orig to make predictions on the dataset with model,
           if different than e_transformers
           x_orig -> x_model
        i_transforms (transformer object or list of transformer objects):
           Transformer(s) needed to make x_orig interpretable
           x_orig -> x_interpret
        interpretable_features (Boolean):
            If True, return explanations using the interpretable feature descriptions instead of
            default names

    Returns:
        Explainer:
            The fitted explainer. Only returned in return_explainer is True
        DataFrame of shape (n_instances, n_features):
            The contribution of each feature. Only returned if return_explanation is True
    """
    if not return_explanation and not return_explainer:
        # TODO: replace with formal warning system
        log.warning(
            "explainer is non-functional with return_contribution and return_explainer set to"
            " false"
        )
        return
    if explainer is None and (model is None or x_train_orig is None or features is None):
        raise ValueError(
            "explainer requires either explainer OR model, x_train, features to be passed"
        )

    if explainer is None:
        explainer = PartialDependenceExplainer(
            model,
            x_train_orig,
            feature_descriptions=feature_descriptions,
            e_transforms=e_transforms,
            m_transforms=m_transforms,
            i_transforms=i_transforms,
            interpretable_features=interpretable_features,
            fit_on_init=True,
        )
    if return_explainer and return_explanation:
        return explainer, explainer.produce()
    if return_explainer:
        return explainer
    if return_explanation:
        return explainer.produce()


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
        super(PartialDependenceExplainer, self).__init__(model, x_train_orig, **kwargs)
        self.base_partial_dependence = PartialDependence(
            model, x_train_orig, features=features, grid_resolution=grid_resolution
        )

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
