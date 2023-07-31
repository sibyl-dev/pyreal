from abc import ABC, abstractmethod

from pyreal.explainers import ExplainerBase


class ExampleBasedBase(ExplainerBase, ABC):
    """
    Base class for Example explainer objects. Abstract class

    Example explainers explain a model or prediction by providing some kind of representative
    example output.

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

    def __init__(self, model, x_train_orig=None, interpretable_features=True, **kwargs):
        self.interpretable_features = interpretable_features
        super(ExampleBasedBase, self).__init__(model, x_train_orig, **kwargs)

    @abstractmethod
    def fit(self, x_train_orig=None, y_train=None):
        """
        Fit this explainer object

        Args:
            x_train_orig (DataFrame of shape (n_instances, n_features):
                Training set to fit on, required if not provided on initialization
            y_train:
                Targets of training set, required if not provided on initialization
        """

    def produce(self, x_orig, n=5):
        """
        Produce the example explanation

        Args:
            x_orig (DataFrame of shape (n_instances, n_features)):
                Input to explain
            n (int):
                Number of examples to return

        Returns:
            ExampleBased
                The explanation
        """
        explanation = self.get_explanation(x_orig, n)
        explanation.update_examples(self.transform_to_x_interpret)
        if self.interpretable_features:
            explanation.update_examples(self.convert_columns_to_interpretable)
        return explanation

    @abstractmethod
    def get_explanation(self, x_orig, n):
        """
        Gets the raw explanation.
        Args:
            x_orig (DataFrame of shape (n_instances, n_features):
                Input to explain
            n (int):
                Number of examples to return

        Returns:
            ExampleBased
        """

    def evaluate_variation(self, with_fit=False, explanations=None, n_iterations=20, n_rows=10):
        """
        Evaluate the variation of the explanations generated by this Explainer.
        A variation of 0 means this explainer is expected to generate the exact same explanation
        given the same model and input. Variation is always non-negative, and can be arbitrarily
        high.

        Args:
            with_fit (Boolean):
                If True, evaluate the variation in explanations including the fit (fit each time
                before running). If False, evaluate the variation in explanations of a pre-fit
                Explainer.
            explanations (None or List of DataFrames of shape (n_instances, n_features)):
                If provided, run the variation check on the precomputed list of explanations
                instead of generating
            n_iterations (int):
                Number of explanations to generate to evaluation variation
            n_rows (int):
                Number of rows of dataset to generate explanations on

        Returns:
            float
                The variation of this Explainer's explanations
        """
        return 0  # TODO: complete this
