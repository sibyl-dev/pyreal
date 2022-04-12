import time
from abc import abstractmethod


class ExplainerChallenge:
    def __init__(self, dataset, evaluations=None, n_rows=10, n_iterations=10):
        self.dataset = dataset
        self.explainer = self.create_explainer()
        if evaluations is None:
            self.evaluations = [
                "fit_time",
                "produce_time",
                "produce_result",
                "series_time",
                "series_result",
                "pre_fit_variation",
                "post_fit_variation",
            ]
        else:
            self.evaluations = evaluations

        self.n_rows = n_rows
        if n_iterations < 2 and self.to_evaluate("variation"):
            raise ValueError("Cannot evaluate variation with fewer than 2 iterations")
        self.n_iterations = n_iterations

    @abstractmethod
    def create_explainer(self):
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
        if self.to_evaluate("produce_time") or self.to_evaluate("produce_result"):
            explanation, produce_time = self.run_challenge_once(
                self.dataset.X.iloc[0 : self.n_rows]
            )
            if self.to_evaluate("produce_time"):
                returns["produce_time"] = produce_time
            if self.to_evaluate("produce_result"):
                returns["produce_result"] = explanation

        # TEST EXPLANATION VARIATION
        if self.to_evaluate("post_fit_variation"):
            variation_score = self.explainer.evaluate_variation(
                with_fit=False, n_iterations=self.n_iterations, n_rows=self.n_rows
            )
            returns["post_fit_variation"] = variation_score

        if self.to_evaluate("pre_fit_variation"):
            variation_score = self.explainer.evaluate_variation(
                with_fit=True, n_iterations=self.n_iterations, n_rows=self.n_rows
            )
            returns["pre_fit_variation"] = variation_score

        return returns

    def to_evaluate(self, name):
        return name in self.evaluations
