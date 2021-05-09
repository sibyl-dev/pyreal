from abc import ABC, abstractmethod
from pyreal.explainers.dte.surrogate_decision_tree import SurrogateDecisionTree
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


def dte(return_explainer=False, explainer=None,
        model=None, x_train_orig=None,
        e_algorithm=None, feature_descriptions=None,
        e_transforms=None, m_transforms=None, i_transforms=None):
    """
    Get a decision tree explanation

    Args:
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
    if explainer is None and (model is None or x_train_orig is None):
        raise ValueError("gfi requires either explainer OR model and x_train to be passed")

    if explainer is None:
        explainer = DecisionTreeExplainer(model, x_train_orig,
                                            e_algorithm=e_algorithm,
                                            feature_descriptions=feature_descriptions,
                                            e_transforms=e_transforms, m_transforms=m_transforms,
                                            i_transforms=i_transforms,
                                            fit_on_init=True)
    # if return_explainer and return_importances:
    #     return explainer, explainer.produce()
    if return_explainer:
        return explainer
    # if return_importances:
    #     return explainer.produce()

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

    def __init__(self, model, x_train_orig, e_algorithm=None, **kwargs):
        if e_algorithm is None:
            e_algorithm = choose_algorithm()
        if e_algorithm == "surrogate_tree":
            self.base_decision_tree = SurrogateDecisionTree(model, x_train_orig, **kwargs)
        if self.base_decision_tree is None:
            raise ValueError("Invalid algorithm type %s" % e_algorithm)

        super(DecisionTreeExplainer, self).__init__(self.base_decision_tree.algorithm, model, x_train_orig, **kwargs)

    def fit(self):
        """
        Fit this explainer object
        """
        self.base_decision_tree.fit()
