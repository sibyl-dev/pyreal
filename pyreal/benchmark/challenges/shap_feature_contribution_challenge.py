import numpy as np

from pyreal.benchmark.challenges import explainer_challenge
from pyreal.explainers import ShapFeatureContribution


class LocalFeatureContributionChallenge(explainer_challenge):
    def create_explainer(self):
        return ShapFeatureContribution(model=self.dataset.model, x_orig=self.dataset.X,
                                       transforms=self.dataset.transforms, fit_on_init=True)

    def evaluate_consistency(self, results):
        # TODO: consider alternative evaluation approaches
        return np.max(np.var(results, axis=0))
