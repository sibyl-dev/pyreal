import numpy as np

from pyreal.explainers.example.base import ExampleBasedBase
from pyreal.explanation_types.explanations.example_based import CounterfactualExplanation
from scipy.optimize import minimize
from alibi.explainers import Counterfactual


class Counterfactuals(ExampleBasedBase):
    """
    A Counterfactuals object computes a similar input to the one given that gives a different model
    prediction.

    Args:
        model (string filepath or model object):
           Filepath to the pickled model to explain, or model object with .predict() function
        x_train_orig (DataFrame of size (n_instances, n_features)):
            Training set in original form.
        **kwargs: see base Explainer args
    """

    def __init__(self, model, x_train_orig=None, **kwargs):
        self.explainer = None
        self.cf = None
        super(Counterfactuals, self).__init__(model, x_train_orig, **kwargs)

    def fit(self, x_train_orig=None, y_train=None):
        """
        Fit the explainer

        Args:
            x_train_orig (DataFrame of shape (n_instances, n_features):
                Training set to fit on, required if not provided on initialization
            y_train:
                Targets of training set, required if not provided on initialization
        """
        return self

    def get_explanation(self, x_orig, target_prediction, num_examples=3):
        """
        Get num_examples counterfactual examples for x_orig

        Args:
            x_orig (DataFrame of shape (n_instances, n_features)):
               The input to be explained
            target_prediction (int or float):
                The prediction of the desired counterfactual example
            num_examples (int):
                Number of neighbors to return
        Returns:
            CounterfactualExplanation
        """
        x_algo = self.transform_to_x_algorithm(x_orig)

        def dist(a, b):
            return np.linalg.norm(a, b)

        def objective_function(x_mod):
            # Difference between the prediction on x_mod and the target prediction:
            o1 = (self.model.predict(x_mod) - target_prediction)**2
            # Distance between x_mod and the input:
            o2 = dist(x_algo, x_mod)
            # Sparsity of changes:
            o3 = np.sum(x_mod != x_algo)
            # Likelihood of features:
            




