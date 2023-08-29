import numpy as np

from pyreal.explainers import ExplainerBase
from pyreal.sample_applications import ames_housing
import time
from pyreal.explanation_types.explanations.example_based import CounterfactualExplanation
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2, RankAndCrowdingSurvival
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.dnsga2 import DNSGA2
from pymoo.core.mixed import (
    MixedVariableGA,
    MixedVariableMating,
    MixedVariableSampling,
    MixedVariableDuplicateElimination,
)
from pymoo.core.problem import ElementwiseProblem, Problem
from pandas.api.types import is_numeric_dtype, is_bool_dtype, is_integer_dtype
from pymoo.core import variable
import math
import pandas as pd


def _dist(a, b):
    return np.linalg.norm(b - a)


class CFProblem(ElementwiseProblem):
    def __init__(self, length, model, x_algo, target_prediction, vars, transform_func):
        self.model = model
        self.target_prediction = target_prediction
        self.column_order = x_algo.columns
        self.x_algo = x_algo.to_numpy().astype(object)
        self.x_model = transform_func(x_algo).to_numpy()
        self.transform_func = transform_func
        super().__init__(n_var=length, n_obj=3, vars=vars)

    def _evaluate(self, x, out, *args, **kwargs):
        x = self.transform_func(pd.DataFrame(x, index=[0])[self.column_order])
        x_np = x.to_numpy()
        # Difference between the prediction on x_mod and the target prediction:
        o1 = abs(self.model.predict(x_np)[0] - self.target_prediction)
        # Distance between x_mod and the input:
        o2 = _dist(self.x_model, x_np)
        # Sparsity of changes:
        o3 = np.sum(self.x_model != x_np)
        # TODO Likelihood of features:
        # o4 = 0
        out["F"] = [o1, o2, o3]


class Counterfactuals(ExplainerBase):
    """
    A Counterfactuals object computes a similar input to the one given that gives a different model
    prediction.

    Args:
        model (string filepath or model object):
           Filepath to the pickled model to explain, or model object with .predict() function
        x_train_orig (DataFrame of size (n_instances, n_features)):
            Training set in original form.
        population_size (int):
            Population size to use for optimization
        num_generations (int):
            Number of generations to use for optimization
        **kwargs: see base Explainer args
    """

    def __init__(
        self, model, x_train_orig=None, population_size=30, num_generations=500, **kwargs
    ):
        self.cf = None
        self.vars = None
        self.population_size = population_size
        self.num_generations = num_generations
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

        self.vars = dict()
        for col in x_train_algo:
            if is_bool_dtype(x_train_algo[col]):
                self.vars[col] = variable.Binary()
            elif is_integer_dtype(x_train_algo[col]):
                self.vars[col] = variable.Integer(
                    bounds=(
                        math.floor(min(x_train_algo[col]) - 0.1 * abs(min(x_train_algo[col]))),
                        math.ceil(max(x_train_algo[col]) + 0.1 * abs(max(x_train_algo[col]))),
                    )
                )
            elif is_numeric_dtype(x_train_algo[col]):
                self.vars[col] = variable.Real(
                    bounds=(
                        min(x_train_algo[col]) - 0.1 * abs(min(x_train_algo[col])),
                        max(x_train_algo[col]) + 0.1 * abs(max(x_train_algo[col])),
                    )
                )
            else:
                self.vars[col] = variable.Choice(options=list(np.unique(x_train_algo[col])))
        return self

    def produce_explanation(self, x_orig, target_prediction=None, num_examples=3):
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
        if target_prediction is None:
            raise NotImplementedError(
                "Counterfactuals currently expects a target prediction be manually specified -"
                " this will be updated in the future."
            )
        if self.vars is None:
            raise AttributeError("Must call fit() before produce()")
        x_algo = self.transform_to_x_algorithm(x_orig)
        problem = CFProblem(
            x_algo.shape[-1],
            self.model,
            x_algo,
            target_prediction,
            vars=self.vars,
            transform_func=self.transform_x_from_algorithm_to_model,
        )
        algorithm = NSGA2(
            pop_size=self.population_size,
            sampling=MixedVariableSampling(),
            mating=MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination()),
            eliminate_duplicates=MixedVariableDuplicateElimination(),
        )
        result = minimize(
            problem, algorithm, termination=("n_gen", self.num_generations), verbose=False
        )
        raw_explanation = dict()
        raw_explanation[0] = pd.DataFrame(list(result.X))[x_algo.columns]
        order = np.argsort(result.F[:, 0])[0:num_examples]
        raw_explanation[0] = pd.DataFrame(list(result.X)).iloc[order][x_algo.columns]
        print(self.model_predict_on_algorithm(raw_explanation[0]))
        return CounterfactualExplanation((raw_explanation, None))

    def evaluate_variation(self, with_fit=False, explanations=None, n_iterations=20, n_rows=10):
        """
        Evaluate the variation of the explanations generated by this Explainer.
        A variation of 0 means this explainer is expected to generate the exact same explanation
        given the same model and input. Variation is always non-negative, and can be arbitrarily
        high.

        Args:
            with_fit (Boolean):
                If True, evaluate the variation in explanations including the fit (fit each time
                before running). If False, evaluate the variation in explanations of a pre-fit
                Explainer.
            explanations (None or List of DataFrames of shape (n_instances, n_features)):
                If provided, run the variation check on the precomputed list of explanations
                instead of generating
            n_iterations (int):
                Number of explanations to generate to evaluation variation
            n_rows (int):
                Number of rows of dataset to generate explanations on

        Returns:
            float
                The variation of this Explainer's explanations
        """
        return 0  # TODO: complete this


start = time.time()
x_train_orig = ames_housing.load_data().drop(columns="Id")
model = ames_housing.load_model()
transformers = ames_housing.load_transformers()

explainer = Counterfactuals(
    model, x_train_orig, transformers=transformers, fit_on_init=True, population_size=30
)
exp = explainer.produce(x_train_orig.iloc[1:2], target_prediction=200000)
print("Total time:", time.time() - start)
print(explainer.model_predict(exp.get_examples()))
