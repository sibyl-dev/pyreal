from pyreal.explainers.llm.narrative_explainer import LocalNarrativeBase
from pyreal.explainers.lfc.local_feature_contribution import LocalFeatureContribution
from pyreal.utils.explanation_utils import get_top_contributors
from openai import OpenAI


def parse_explanation_to_input(explanation):
    """
    Parse a narrative explanation to input format
    Args:
        explanation (Series of strings):
            Narrative explanation

    Returns:
        DataFrame of shape (n_instances, n_features)
            Feature contribution explanations. Each row represents an instance, and each
            column a feature.
    """
    print(explanation)


class NarrativeLocalFeatureContribution(LocalNarrativeBase):
    """
    NarrativeLocalFeatureContribution object.

    A NarrativeLocalFeatureContribution object generates local feature contribution explanations
    in a narrative format.
    """

    def __init__(
        self,
        model,
        x_train_orig=None,
        y_train=None,
        transformers=None,
        wrapped_explainer_kwargs=None,
        **kwargs
    ):
        """
        Initialize a NarrativeLocalFeatureContribution object.
        Args:
            model (string filepath or model object):
                Filepath to the pickled model to explain, or model object with .predict() function
            x_train_orig (DataFrame of size (n_instances, n_features)):
                Training set in original form.
            wrapped_explainer_kwargs (dict):
                Keyword arguments to pass to the wrapped explainer
            **kwargs: See LocalNarrativeBase args
        """
        if wrapped_explainer_kwargs is None:
            wrapped_explainer_kwargs = {}
        self.wrapped_explainer = LocalFeatureContribution(
            model,
            x_train_orig,
            y_train=y_train,
            transformers=transformers,
            **wrapped_explainer_kwargs
        )
        super(NarrativeLocalFeatureContribution, self).__init__(model, x_train_orig, **kwargs)

    def narrify(self, explanation, num_features=None):
        """
        Generate a narrative explanation from a feature contribution explanation
        Args:
            explanation (DataFrame of shape (n_instances, n_features)):
                Feature contribution explanations. Each row represents an instance, and each
                column a feature.
            num_features (int):
                Number of features to include in the explanation.

        Returns:
            Series of strings
                One narrative explanation per instance
        """
        print(
            explanation
        )  # extract top contributors from explanation. Can't use get_top_contributors, as that expects the realapp output

    def create_base_explainer(self, model, x_train_orig, **kwargs):
        return LocalFeatureContribution(model, x_train_orig, **kwargs)


from pyreal.sample_applications import ames_housing

X_train, y_train = ames_housing.load_data(include_targets=True)
explainer = NarrativeLocalFeatureContribution(
    model=ames_housing.load_model(),
    x_train_orig=X_train,
    y_train=y_train,
    transformers=ames_housing.load_transformers(),
)
explainer.fit(X_train, y_train)
print(explainer.produce_explanation_interpret(X_train.iloc[0:2]))
