import numpy as np

from pyreal.explainers.example.base import ExampleBasedBase
from pyreal.explanation_types.explanations.example_based import CounterfactualExplanation
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2, RankAndCrowdingSurvival
from pymoo.core.mixed import MixedVariableGA
from pymoo.core.problem import ElementwiseProblem, Problem
from pandas.api.types import is_numeric_dtype, is_bool_dtype, is_integer_dtype
from pymoo.core import variable
import math
import pandas as pd


def _dist(a, b):
    return np.sum((b - a) ** 2)


class CFProblem(ElementwiseProblem):
    def __init__(self, length, model, x_algo, target_prediction, vars):
        self.model = model
        self.target_prediction = target_prediction
        self.x_algo = x_algo.squeeze()
        super().__init__(n_var=length, n_obj=4, vars=vars)

    def _evaluate(self, x, out, *args, **kwargs):
        x = pd.DataFrame(x, index=[0])[self.x_algo.index]
        # Difference between the prediction on x_mod and the target prediction:
        o1 = np.sum((self.model.predict(x) - self.target_prediction) ** 2)
        # Distance between x_mod and the input:
        o2 = _dist(self.x_algo, x.squeeze())
        # Sparsity of changes:
        o3 = np.sum(self.x_algo != x.squeeze())
        # TODO Likelihood of features:
        o4 = 0
        out["F"] = [o1, o2, o3, o4]


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
        x_train_algo = self.transform_to_x_algorithm(self._get_x_train_orig(x_train_orig))
        # self.xl = self.transform_to_x_model(x_train_orig).min(axis=0) * 1.2
        # self.xu = self.transform_to_x_model(x_train_orig).max(axis=0) * 1.2

        self.vars = dict()
        for col in x_train_algo:
            if is_bool_dtype(x_train_algo[col]):
                self.vars[col] = variable.Binary()
            elif is_integer_dtype(x_train_algo[col]):
                self.vars[col] = variable.Integer(
                    bounds=(
                        math.floor(min(x_train_algo[col]) * 1.2),
                        math.ceil(max(x_train_algo[col]) * 1.2),
                    )
                )
            elif is_numeric_dtype(x_train_algo[col]):
                self.vars[col] = variable.Real(
                    bounds=(min(x_train_algo[col]) * 1.2, max(x_train_algo[col]) * 1.2)
                )
            else:
                self.vars[col] = variable.Choice(options=np.unique(x_train_algo[col]))

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
        if self.vars is None:
            raise AttributeError("Must call fit() before produce()")
        x_algo = self.transform_to_x_algorithm(x_orig)
        problem = CFProblem(
            x_algo.shape[-1], self.model, x_algo, target_prediction, vars=self.vars
        )
        algorithm = MixedVariableGA(pop_size=100, survival=RankAndCrowdingSurvival())
        result = minimize(problem, algorithm, ("n_gen", 200), seed=1)


from pyreal.sample_applications import ames_housing
from pyreal.transformers import MultiTypeImputer

x_train_orig = ames_housing.load_data(n_rows=100).drop(columns="Id")
model = ames_housing.load_model()
transformers = ames_housing.load_transformers()
# transformers.append(MultiTypeImputer(algorithm=True, model=True).fit(x_train_orig))

explainer = Counterfactuals(model, x_train_orig, transformers=transformers, fit_on_init=True)
explainer.get_explanation(x_train_orig.iloc[0], target_prediction=200000)
