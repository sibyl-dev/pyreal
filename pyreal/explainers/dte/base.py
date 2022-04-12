from abc import ABC, abstractmethod

from pyreal.explainers import ExplainerBase


class DecisionTreeExplainerBase(ExplainerBase, ABC):
    """
    Base class for DecisionTree explainer objects. Abstract class

    A DecisionTreeExplainerBase object explains by 1) decomposing arbitrary models into
    decision trees and 2) visualizing the decision rules associated with a prediction.

    Args:
        model (string filepath or model object):
           Filepath to the pickled model to explain, or model object with .predict() function
        x_train_orig (dataframe of shape (n_instances, x_orig_feature_count)):
           The training set for the explainer
        interpretable_features (Boolean):
            If True, return explanations using the interpretable feature descriptions instead of
            default names
        **kwargs: see base Explainer args
    """

    def __init__(self, model, x_train_orig, interpretable_features=True, **kwargs):
        self.interpretable_features = interpretable_features
        super(DecisionTreeExplainerBase, self).__init__(model, x_train_orig, **kwargs)

    @abstractmethod
    def fit(self):
        """
        Fit this explainer object
        """

    @abstractmethod
    def produce(self, x_orig=None):
        """
        Produce the decision tree explanation

        Args:
            x_orig (None):
                Decision tree explanations do not take inputs - dummy to match signature

        Returns:
            A decision tree model
        """

    def return_features(self):
        """
        Returns the (interpreted) features of the dataset.

        Returns:
            The features of the dataset. Interpret the features if
            `interpretable_features` is set to true.
        """
        x_algorithm = self.transform_to_x_algorithm(self._x_train_orig)

        if self.interpretable_features:
            features = self.convert_columns_to_interpretable(x_algorithm).columns
        else:
            features = x_algorithm.columns

        return features

    def evaluate_variation(self, with_fit=False, explanations=None, n_iterations=20, n_rows=10):
        """
        Not currently implemented for decision tree explainers

        Args:
            with_fit (Boolean):
                If True, evaluate the variation in explanations including the fit (fit each time
                before running). If False, evaluate the variation in explanations of a pre-fit
                Explainer.
            explanations (None or List of Explanation Objects):
                If provided, run the variation check on the precomputed list of explanations
                instead of generating
            n_iterations (int):
                Number of explanations to generate to evaluation variation
            n_rows (int):
                Number of rows of dataset to generate explanations on

        Returns:
            None

        Raises:
            NotImplementedError
        """
        raise NotImplementedError
