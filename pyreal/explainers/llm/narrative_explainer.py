from pyreal.explainers.base import ExplainerBase
from abc import abstractmethod, ABC
from openai import OpenAI


class LocalNarrativeBase(ExplainerBase, ABC):
    def __init__(self, openai_key, base_explainer, **kwargs):
        """
        Generates explanations in a narrative format, based on one an explanation generated
        using other techniques.

        Args:
            model (string filepath or model object):
                Filepath to the pickled model to explain, or model object with .predict() function
            openai_key (string):
                OpenAI GPT API key
            x_train_orig (DataFrame of size (n_instances, n_features)):
                Training set in original form.
            **kwargs: See ExplainerBase args
        """
        self.wrapped_explainer = base_explainer
        self.client = OpenAI(api_key=openai_key)
        super(ExplainerBase, self).__init__(model, x_train_orig, **kwargs)

    def fit(self, x_train_orig=None, y_train=None):
        self.wrapped_explainer.fit(x_train_orig, y_train)

    def produce_explanation_interpret(self, x_orig, num_features=None, **kwargs):
        """
        Generate a narrative explanation

        Args:
            x_orig (DataFrame of shape (n_instances, n_features)):
                Instances to explain
            num_features (int):
                Number of features to include in the explanation. If None, include all features.
            **kwargs: see base Explainer args
        """
        return self.narrify(
            self.wrapped_explainer.produce_explanation(x_orig, **kwargs), num_features=num_features
        )

    @abstractmethod
    def narrify(self, explanation, num_features=None):
        """
        Generate a narrative explanation from a base explanation

        Args:
            explanation:
                Base explanation to turn into a narrative
            num_features (int):
                Number of features to include in the explanation.
        """

    @abstractmethod
    def create_base_explainer(self, model, x_train_orig, **kwargs):
        """
        Create the base explainer to wrap
        """

    def evaluate_variation(self, with_fit=False, explanations=None, n_iterations=20, n_rows=10):
        return 0

    def produce_explanation(self, x_orig, **kwargs):
        """
        Unused for narrative explainers as explanations are directly produced in the
        interpretable feature space
        """
        return None
