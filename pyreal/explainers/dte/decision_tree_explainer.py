from abc import ABC, abstractmethod
from pyreal.explainers import DecisionTreeExplainerBase

def choose_algorithm():
    """
    Choose an algorithm based on the model type.
    Currently, shap is the only supported algorithm

    Return:
        string (one of ["surrogate_tree"])
            Explanation algorithm to use
    """
    return "surrogate_tree"


def dte(return_importances=True, return_explainer=False, explainer=None,
        model=None, x_train_orig=None,
        e_algorithm=None, feature_descriptions=None,
        e_transforms=None, m_transforms=None, i_transforms=None,
        interpretable_features=True):
    """
    Get a decision tree explanation

    Args:
        return_importances (Boolean):
            If true, return explanation of features importance.
            If true, requires one of `explainer` or (`model and x_train`)
        return_explainer (Boolean):
            If true, return the fitted Explainer object.
            If true, requires one of `explainer` or (`model and x_train`)
        explainer (Explainer):
            Fitted explainer object.
        model (string filepath or model object):
           Filepath to the pickled model to explain, or model object with .predict() function
        x_train_orig (dataframe of shape (n_instances, x_orig_feature_count)):
           The training set for the explainer
        e_algorithm (string, one of ["shap"]):
           Explanation algorithm to use. If none, one will be chosen automatically based on model
           type
        feature_descriptions (dict):
           Interpretable descriptions of each feature
        e_transforms (transformer object or list of transformer objects):
           Transformer(s) that need to be used on x_orig for the explanation algorithm:
           x_orig -> x_explain
        m_transforms (transformer object or list of transformer objects):
           Transformer(s) needed on x_orig to make predictions on the dataset with model,
           if different than e_transforms
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
            The importance of each feature. Only returned if return_importance is True
    """
    if not return_importances and not return_explainer:
        # TODO: replace with formal warning system
        raise ValueError("dte is non-functional with return_importances and return_explainer set to false")
    if explainer is None and (model is None or x_train_orig is None):
        raise ValueError("gfi requires either explainer OR model and x_train to be passed")

    if explainer is None:
        explainer = DecisionTreeExplainer(model, x_train_orig,
                                            e_algorithm=e_algorithm,
                                            feature_descriptions=feature_descriptions,
                                            e_transforms=e_transforms, m_transforms=m_transforms,
                                            i_transforms=i_transforms,
                                            interpretable_features=interpretable_features,
                                            fit_on_init=True)
    if return_explainer and return_importances:
        return explainer, explainer.produce()
    if return_explainer:
        return explainer
    if return_importances:
        return explainer.produce()

class DecisionTreeExplainer(DecisionTreeExplainerBase):
    """
    Generic DecisionTreeExplainer wrapper

    An DecisionTreeExplainer object wraps multiple global feature-based explanations. If no
    specific algorithm is requested, one will be chosen based on the information given.
    Currently, only surrogate tree is supported.

    Args:
        model (string filepath or model object):
           Filepath to the pickled model to explain, or model object with .predict() function
        x_train_orig (dataframe of shape (n_instances, x_orig_feature_count)):
           The training set for the explainer
        e_algorithm (string, one of ["surrogate_tree"]):
           Explanation algorithm to use. If none, one will be chosen automatically based on model
           type
        **kwargs: see DecisionTreeExplainerBase args
    """

    def __init__(self, algorithm, model, x_train_orig, interpretable_features=True, **kwargs):
        self.interpretable_features = interpretable_features
        self.importance = None
        super(DecisionTreeExplainerBase, self).__init__(algorithm, model, x_train_orig, **kwargs)

    @abstractmethod
    def fit(self):
        """
        Fit this explainer object
        """

    @abstractmethod
    def produce(self, x_orig=None):
        """
        Produce a decision tree
        """
