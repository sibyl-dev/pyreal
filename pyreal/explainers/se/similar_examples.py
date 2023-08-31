import pandas as pd
from sklearn.neighbors import KDTree
from sklearn.preprocessing import StandardScaler

from pyreal.explainers.se.base import SimilarExamplesBase
from pyreal.explanation_types.explanations.example_based import SimilarExampleExplanation


class SimilarExamples(SimilarExamplesBase):
    """
    SimilarExamples object.

    A SimilarExamples object gets feature explanation using the Nearest Neighbors algorithm

    SimilarExamples explainers expect data to be entirely numeric

    Args:
        model (string filepath or model object):
           Filepath to the pickled model to explain, or model object with .predict() function
        x_train_orig (DataFrame of size (n_instances, n_features)):
            Training set in original form.
        standardize (Boolean):
            If True, standardize the data when selected similar examples
        **kwargs: see base Explainer args
    """

    def __init__(self, model, x_train_orig=None, standardize=False, **kwargs):
        self.explainer = None
        self.standardize = standardize
        self.standardizer = None
        super(SimilarExamples, self).__init__(model, x_train_orig, **kwargs)

    def fit(self, x_train_orig=None, y_train=None):
        """
        Fit the explainer

        Args:
            x_train_orig (DataFrame of shape (n_instances, n_features):
                Training set to fit on, required if not provided on initialization
            y_train (Series of shape (n_features):
                Targets of training set, required if not provided on initialization
        """
        x_train_orig, y_train = self._get_training_data(x_train_orig, y_train)

        dataset = self.transform_to_x_algorithm(x_train_orig)
        if self.standardize:
            self.standardizer = StandardScaler()
            dataset = self.standardizer.fit_transform(dataset)
        self.explainer = KDTree(dataset)
        self.y_train = y_train
        self.x_train_orig = x_train_orig

        return self

    def produce_explanation_interpret(self, x_orig, disable_feature_descriptions=False, n=5):
        """
        Get the n nearest neighbors to x_orig

        Args:
            x_orig (DataFrame of shape (n_instances, n_features)):
               The input to be explained
            disable_feature_descriptions (Boolean):
                If False, do not apply feature descriptions
            n (int):
                Number of neighbors to return
        Returns:
            SimilarExamplesExplanation
                Set of similar examples and their targets
        """
        if self.explainer is None:
            raise AttributeError("Instance has no explainer. Must call fit() before produce()")
        x = self.transform_to_x_algorithm(x_orig)
        if self.standardize:
            x = self.standardizer.transform(x)
        inds = self.explainer.query(x, k=n, return_distance=False)
        raw_explanation_x = {}
        raw_explanation_y = {}
        for i in range(len(inds)):
            raw_explanation_x[i] = self.x_train_orig.iloc[inds[i], :]
            raw_explanation_y[i] = pd.Series(self.y_train.iloc[inds[i]].squeeze())
        x_interpret = self.transform_to_x_interpret(x_orig)
        explanation = SimilarExampleExplanation(
            (raw_explanation_x, raw_explanation_y), x_interpret
        )
        explanation.update_examples(self.transform_to_x_interpret)
        return explanation

    def produce_explanation(self, x_orig, **kwargs):
        """
        Unused for similar examples explainers as explanations are directly produced in the
        interpretable feature space
        """
        return None
