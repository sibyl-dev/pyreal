import time
from abc import abstractmethod

import numpy as np


class ExplainerChallenge:
    def __init__(self, dataset, evaluations=None, n_rows=10, n_iterations=10):
        self.dataset = dataset
        self.explainer = self.create_explainer()
        if evaluations is None:
            self.evaluations = ["fit_time", "produce_time", "produce_result",
                                "series_time", "series_result", "pre_fit_consistency",
                                "post_fit_consistency"]
        else:
            self.evaluations = evaluations

        self.n_rows = n_rows
        if n_iterations < 2 and self.to_evaluate("consistency"):
            raise ValueError("Cannot evaluate consistency with fewer than 2 iterations")
        self.n_iterations = n_iterations

    @abstractmethod
    def create_explainer(self):
        pass

    @abstractmethod
    def evaluate_consistency(self, results):
        pass

    def run_fit(self):
        fit_start_time = time.time()
        self.explainer.fit()
        fit_end_time = time.time()
        return fit_end_time - fit_start_time

    def run_challenge_once(self, x):
        produce_start_time = time.time()
        explanation = self.explainer.produce(x)
        produce_end_time = time.time()
        return explanation, produce_end_time - produce_start_time

    def run(self):
        returns = {}

        # FIT EXPLAINER AND TIME
        fit_time = self.run_fit()
        if self.to_evaluate("fit_time"):
            returns["fit_time"] = fit_time

        # TEST ON A SINGLE SERIES
        if self.to_evaluate("series_time") or self.to_evaluate("series_result"):
            explanation_series, time_series = self.run_challenge_once(self.dataset.X.iloc[0])
            if self.to_evaluate("series_time"):
                returns["series_time"] = time_series
            if self.to_evaluate("series_result"):
                returns["series_result"] = explanation_series

        # TEST ON N_ROWS ITEMS
        if self.to_evaluate("produce_time") or self.to_evaluate("produce_result") \
                or self.to_evaluate("pre_fit_consistency") \
                or self.to_evaluate("post_fit_consistency"):
            explanation, produce_time = self.run_challenge_once(self.dataset.X.iloc[0:self.n_rows])
            if self.to_evaluate("produce_time"):
                returns["produce_time"] = produce_time
            if self.to_evaluate("produce_result"):
                returns["produce_result"] = explanation

            # TEST EXPLANATION CONSISTENCY
            if self.to_evaluate("post_fit_consistency"):
                results = [explanation.to_numpy()]
                for i in range(self.n_iterations - 1):
                    results.append(
                        self.run_challenge_once(self.dataset.X.iloc[0:self.n_rows])[0].to_numpy())
                consistency_score = self.evaluate_consistency(np.array(results))
                returns["post_fit_consistency"] = consistency_score

            if self.to_evaluate("pre_fit_consistency"):
                results = [explanation.to_numpy()]
                for i in range(self.n_iterations - 1):
                    self.run_fit()
                    results.append(
                        self.run_challenge_once(self.dataset.X.iloc[0:self.n_rows])[0].to_numpy())
                consistency_score = self.evaluate_consistency(np.array(results))
                returns["pre_fit_consistency"] = consistency_score

        return returns

    def to_evaluate(self, name):
        return name in self.evaluations