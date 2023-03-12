from pyreal.benchmark.challenges.explainer_challenge import ExplainerChallenge
from pyreal.explainers import ShapFeatureImportance


class ShapFeatureImportanceChallenge(ExplainerChallenge):
    def create_explainer(self):
        return ShapFeatureImportance(
            model=self.dataset.model,
            x_train_orig=self.dataset.X,
            transformers=self.dataset.transforms,
            fit_on_init=True,
        )
