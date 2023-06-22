from sklearn.neighbors import KDTree

from pyreal.explainers.example.base import ExampleBasedBase
from pyreal.types.explanations.example_based import SimilarExampleExplanation


class SimilarExamples(ExampleBasedBase):
    """
    SimilarExamples object.

    A SimilarExamples object gets feature explanation using the Nearest Neighbors algorithm

    SimilarExamples explainers expect data to be entirely numeric

    Args:
        model (string filepath or model object):
           Filepath to the pickled model to explain, or model object with .predict() function
        x_train_orig (DataFrame of size (n_instances, n_features)):
            Training set in original form.
        **kwargs: see base Explainer args
    """

    def __init__(self, model, x_train_orig=None, **kwargs):
        self.explainer = None
        super(SimilarExamples, self).__init__(model, x_train_orig, **kwargs)

    def fit(self, x_train_orig=None, y_train=None):
        """
        Fit the explainer

        Args:
            x_train_orig (DataFrame of shape (n_instances, n_features):
                Training set to fit on, required if not provided on initialization
            y_train:
                Targets of training set, required if not provided on initialization
        """
        x_train_orig, y_train = self._get_training_data(x_train_orig, y_train)

        dataset = self.transform_to_x_algorithm(x_train_orig)
        self.explainer = KDTree(dataset)
        self.y_train = y_train
        self.x_train_orig = x_train_orig

        return self

    def get_explanation(self, x_orig, n=5):
        """
        Get the n nearest neighbors to x_orig

        Args:
            x_orig (DataFrame of shape (n_instances, n_features)):
               The input to be explained
            n (int):
                Number of neighbors to return
        Returns:
            SimilarExamples
        """
        if self.explainer is None:
            raise AttributeError("Instance has no explainer. Must call fit() before produce()")
        x = self.transform_to_x_algorithm(x_orig)

        inds = self.explainer.query(x, k=n, return_distance=False)
        explanation = {}
        for i in range(len(inds)):
            explanation[i] = (self.x_train_orig.iloc[inds[i], :], self.y_train.iloc[inds[i]])
        return SimilarExampleExplanation(explanation)
