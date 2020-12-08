import numpy as np

from pyreal.benchmark.challenges.explainer_challenge import ExplainerChallenge
from pyreal.explainers import ShapFeatureContribution


class ShapFeatureContributionChallenge(ExplainerChallenge):
    def create_explainer(self):
        return ShapFeatureContribution(model=self.dataset.model, x_orig=self.dataset.X,
                                       transforms=self.dataset.transforms, fit_on_init=True,
                                       shap_type="linear")

    def evaluate_consistency(self, results):
        # TODO: consider alternative evaluation approaches
        return np.max(np.var(results, axis=0))
