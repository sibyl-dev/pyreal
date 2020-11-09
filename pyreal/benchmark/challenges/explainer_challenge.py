import time
from abc import abstractmethod

import numpy as np


class ExplainerChallenge:
    def __init__(self, dataset, evaluations=None, n_rows=10, n_iterations=10):
        self.dataset = dataset
        self.explainer = self.create_explainer(dataset)
        if evaluations is None:
            self.evaluations = ["fit_time", "produce", "consistency", "series"]
        else:
            self.evaluations = evaluations

        self.n_rows = n_rows
        if n_iterations < 2:
            raise ValueError("Cannot evaluate consistency with fewer than 2 iterations")
        self.n_iterations = n_iterations

    @abstractmethod
    def create_explainer(self, dataset):
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
        if self.evaluate("fit_time"):
            returns["fit_time"] = fit_time

        # TEST ON A SINGLE SERIES
        if self.evaluate("series"):
            explanation_series, time_series = self.run_challenge_once(self.dataset.X.iloc[0])
            returns["series"] = {"explanation": explanation_series, "time": time_series}

        # TEST ON N_ROWS ITEMS
        if self.evaluate("produce") or self.evaluate("consistency"):
            explanation, time = self.run_challenge_once(self.dataset.X.iloc[0:self.n_rows])
            if self.evaluate("produce"):
                returns["produce"] = {"explanation": explanation, "time": time}

            # TEST EXPLANATION CONSISTENCY
            if self.evaluate("consistency"):
                results = [explanation.to_numpy()]
                for i in range(self.n_iterations - 1):
                    results.append(
                        self.run_challenge_once(self.dataset.X.iloc[0:self.n_rows])[0].to_numpy())
                consistency_score = self.evaluate_consistency(np.array(results))
                returns["consistency"] = consistency_score

        return returns

    def evaluate(self, name):
        return name in self.evaluations
