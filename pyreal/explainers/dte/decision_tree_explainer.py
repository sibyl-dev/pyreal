from abc import ABC, abstractmethod
from pyreal.explainers import DecisionTreeExplainerBase

def choose_algorithm():
    """
    Choose an algorithm based on the model type.
    Currently, shap is the only supported algorithm

    Return:
        string (one of ["shap"])
            Explanation algorithm to use
    """
    return "surrogate_tree"

def dte():

    return DecisionTreeExplainer()

class DecisionTreeExplainer(DecisionTreeExplainerBase):
    """
    Base class for DecisionTree explainer objects. Abstract class

    A DecisionTreeExplainerBase object explains by 1) decomposing arbitrary models into decision trees and 2) visualizing the decision rules associated with a prediction.

    Args:
        algorithm (ExplanationAlgorithm or None):
            Name of the algorithm this Explainer uses
        model (string filepath or model object):
           Filepath to the pickled model to explain, or model object with .predict() function
        x_train_orig (dataframe of shape (n_instances, x_orig_feature_count)):
           The training set for the explainer
        e_algorithm (string, one of ["shap"]):
           Explanation algorithm to use. If none, one will be chosen automatically based on model
           type
        interpretable_features (Boolean):
            If True, return explanations using the interpretable feature descriptions instead of
            default names
        **kwargs: see base Explainer args
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
