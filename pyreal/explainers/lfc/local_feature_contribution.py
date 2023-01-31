import logging

from pyreal.explainers import (
    LocalFeatureContributionsBase,
    ShapFeatureContribution,
    SimpleCounterfactualContribution,
)

log = logging.getLogger(__name__)


def choose_algorithm():
    """
    Choose an algorithm based on the model type.
    Currently, shap is the only supported algorithm

    Returns:
        string (one of ["shap"])
            Explanation algorithm to use
    """
    return "shap"


def lfc(
    return_contributions=True,
    return_explainer=False,
    explainer=None,
    model=None,
    x_orig=None,
    x_train_orig=None,
    e_algorithm=None,
    feature_descriptions=None,
    e_transforms=None,
    m_transforms=None,
    i_transforms=None,
    interpretable_features=True,
):
    """
    Get a local feature contribution for x_input

    Args:
        return_contributions (Boolean):
            If true, return explanation of features in x_input.
            If true, requires `x_input` and one of `explainer` or (`model and x_train`)
        return_explainer (Boolean):
            If true, return the fitted Explainer object.
            If true, requires one of `explainer` or (`model and x_train`)
        explainer (Explainer):
            Fitted explainer object.
        model (string filepath or model object):
           Filepath to the pickled model to explain, or model object with .predict() function
        x_orig (dataframe of shape (n_instances, x_orig_feature_count)):
           The input to explain
        x_train_orig (dataframe of shape (n_instances, x_orig_feature_count)):
           The training set for the explainer
        e_algorithm (string, one of ["shap"]):
           Explanation algorithm to use. If none, one will be chosen automatically based on model
           type
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
            The contribution of each feature. Only returned if return_contributions is True
    """
    if not return_contributions and not return_explainer:
        # TODO: replace with formal warning system
        log.warning(
            "lfc is non-functional with return_contribution and return_explainer set to false"
        )
        return
    if explainer is None and (model is None or x_train_orig is None):
        raise ValueError("lfc requires either explainer OR model and x_train to be passed")
    if return_contributions is True and x_orig is None:
        raise ValueError("return_contributions tag require x_input to be passed")

    if explainer is None:
        explainer = LocalFeatureContribution(
            model,
            x_train_orig,
            e_algorithm=e_algorithm,
            feature_descriptions=feature_descriptions,
            e_transforms=e_transforms,
            m_transforms=m_transforms,
            i_transforms=i_transforms,
            interpretable_features=interpretable_features,
            fit_on_init=True,
        )
    if return_explainer and return_contributions:
        return explainer, explainer.produce(x_orig)
    if return_explainer:
        return explainer
    if return_contributions:
        return explainer.produce(x_orig)


class LocalFeatureContribution(LocalFeatureContributionsBase):
    """
    Generic LocalFeatureContribution wrapper

    A LocalFeatureContributions object wraps multiple local feature-based explanations. If no
    specific algorithm is requested, one will be chosen based on the information given.
    Currently, only SHAP is supported.

    Args:
        model (string filepath or model object):
           Filepath to the pickled model to explain, or model object with .predict() function
        x_train_orig (dataframe of shape (n_instances, x_orig_feature_count)):
           The training set for the explainer
        e_algorithm (string, one of ["shap"]):
           Explanation algorithm to use. If none, one will be chosen automatically based on model
           type
        **kwargs: see LocalFeatureContributionsBase args
    """

    def __init__(self, model, x_train_orig, e_algorithm=None, **kwargs):
        if e_algorithm is None:
            e_algorithm = choose_algorithm()
        self.base_local_feature_contribution = None
        if e_algorithm == "shap":
            self.base_local_feature_contribution = ShapFeatureContribution(
                model, x_train_orig, **kwargs
            )
        elif e_algorithm == "simple":
            self.base_local_feature_contribution = SimpleCounterfactualContribution(
                model, x_train_orig, **kwargs
            )
        if self.base_local_feature_contribution is None:
            raise ValueError("Invalid algorithm type %s" % e_algorithm)

        super(LocalFeatureContribution, self).__init__(model, x_train_orig, **kwargs)

    def fit(self):
        """
        Fit this explainer object
        """
        self.base_local_feature_contribution.fit()
        return self

    def get_contributions(self, x_orig):
        """
        Gets the raw explanation.
        Args:
            x_orig (DataFrame of shape (n_instances, n_features):
                Input to explain

        Returns:
            DataFrame of shape (n_instances, n_features)
                Contribution of each feature for each instance
        """
        return self.base_local_feature_contribution.get_contributions(x_orig)
