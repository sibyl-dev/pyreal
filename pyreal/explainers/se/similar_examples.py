import faiss
import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
from sklearn.preprocessing import StandardScaler

from pyreal.explainers.se.base import SimilarExamplesBase
from pyreal.explanation_types import SimilarExampleExplanation
from pyreal.explanation_types.base import convert_columns_with_dict


# From:
# towardsdatascience.com/make-knn-300-times-faster-than-scikit-learns-in-20-lines-5e29d74e76bb
class FaissKNeighbors:
    def __init__(self, X):
        self.index = faiss.IndexFlatL2(X.shape[1])
        self.index.add(X.astype(np.float32))

    def query(self, x, k, return_distance=False):
        distances, indices = self.index.search(x.astype(np.float32), k=k)
        if return_distance:
            return distances, indices
        return indices


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
        fast (Boolean):
            If True, use a faster algorithm to compute the neighbors. Set to False if having
            trouble with faiss library
        **kwargs: see base Explainer args
    """

    def __init__(self, model, x_train_orig=None, standardize=False, fast=True, **kwargs):
        self.explainer = None
        self.standardize = standardize
        self.standardizer = None
        self.x_train_interpret = None
        self.x_train_interpret_features = None
        self.fast = fast
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

        if self.fast:
            self.explainer = FaissKNeighbors(dataset)
        else:
            self.explainer = KDTree(dataset)
        self.y_train = y_train

        self.x_train_interpret = self.transform_to_x_interpret(x_train_orig)
        return self

    def produce_explanation_interpret(
        self, x_orig, disable_feature_descriptions=False, num_examples=5
    ):
        """
        Get the n nearest neighbors to x_orig

        Args:
            x_orig (DataFrame of shape (n_instances, n_features)):
               The input to be explained
            disable_feature_descriptions (Boolean):
                If False, do not apply feature descriptions
            num_examples (int):
                Number of neighbors to return
        Returns:
            SimilarExamplesExplanation
                Set of similar examples and their targets
        """
        if self.explainer is None:
            raise AttributeError("Instance has no explainer. Must call fit() before produce()")
        if not disable_feature_descriptions:  # Running this here for optimization
            x_train_interpret = convert_columns_with_dict(
                self.x_train_interpret, self.feature_descriptions
            )
        else:
            x_train_interpret = self.x_train_interpret
        x = self.transform_to_x_algorithm(x_orig)
        if self.standardize:
            x = self.standardizer.transform(x)
        inds = self.explainer.query(x, k=num_examples, return_distance=False)
        raw_explanation_x = {}
        raw_explanation_y = {}
        for i in range(len(inds)):
            raw_explanation_x[i] = x_train_interpret.iloc[inds[i], :]
            raw_explanation_y[i] = pd.Series(self.y_train.iloc[inds[i]].squeeze())
        x_interpret = self.transform_to_x_interpret(x_orig)
        explanation = SimilarExampleExplanation(
            (raw_explanation_x, raw_explanation_y), x_interpret
        )
        return explanation

    def produce_explanation(self, x_orig, **kwargs):
        """
        Unused for similar examples explainers as explanations are directly produced in the
        interpretable feature space
        """
        return None
