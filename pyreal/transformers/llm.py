import explingo

from pyreal.explanation_types import NarrativeExplanation
from pyreal.transformers import TransformerBase


class NarrativeTransformer(TransformerBase):
    """
    Transforms explanations to narrative (natural-language) form.
    """

    def __init__(
        self,
        llm=None,
        openai_api_key=None,
        num_features=5,
        gpt_model_type="gpt-4o",
        context_description="",
        max_tokens=200,
        training_examples=None,
        **kwargs,
    ):
        """
        Transforms explanations to narrative (natural-language) form.
        Args:
            llm (LLM model object): Local LLM object or LLM client object to use to generate \
                narratives. One of `llm` or `openai_api_key` must be provided.
            openai_api_key (string): OpenAI API key to use
            x_orig (DataFrame of shape (n_instances, n_features):
                Input to explain
            num_features (int):
                Number of features to include in the explanation, when relevant.
                If None, all features will be included
            gpt_model_type (string):
                OpenAI model to use to generate the explanation, if passing in an openai api key
            context_description (string):
                Description of the model's prediction task, in sentence format. This will be
                passed to the LLM and may help produce more accurate explanations.
                For example: "The model predicts the price of houses."
            max_tokens (int):
                Maximum number of tokens to use in the explanation
            training_examples (dictionary of string:list of tuples):
                Few-shot training examples.
                Keys are explanation type (currently support: "feature_contributions")
                Values are lists of tuples, where the first element is the input to the model
                    and the second element is the example output.
                Use the RealApp train_llm functions to populate this
        Returns:
            list of strings of length n_instances
                Narrative version of feature contribution explanation, one item per instance
        """
        self.llm = llm
        self.openai_api_key = openai_api_key
        if self.llm is None and self.openai_api_key is None:
            raise ValueError("Must provide llm or openai_api_key")

        self.narrators = {}

        self.num_features = num_features
        self.llm_model = gpt_model_type
        self.context_description = context_description
        self.max_tokens = max_tokens

        if training_examples is not None:
            self.training_examples = training_examples
        else:
            self.training_examples = {}

        if "interpret" not in kwargs:
            kwargs["interpret"] = True
        if "model" not in kwargs:
            kwargs["model"] = False

        super().__init__(require_values=True, **kwargs)

    def data_transform(self, x):
        return x

    def fit(self, x, **params):
        return self

    def set_training_examples(self, explanation_type, training_examples, replace=False):
        """
        Set examples of narrative explanations for the request explanation type.

        Args:
            explanation_type (string):
                Type of explanation to set examples for. Currently only "feature_contributions"
                is supported.
            training_examples (list of tuples):
                List of tuples, where the first element is the input to the model
                and the second element is the example output.
            replace (bool):
                If True, replace existing examples. If False, append to existing examples.
        """
        if explanation_type not in ["feature_contributions"]:
            raise ValueError(
                "Invalid training example type %s. Expected one of ['feature_contributions']"
                % explanation_type
            )
        if replace:
            self.training_examples[explanation_type] = training_examples
        else:
            if self.training_examples.get(explanation_type) is None:
                self.training_examples[explanation_type] = []
            self.training_examples[explanation_type].extend(training_examples)

    def transform_explanation_feature_contribution(self, explanation, num_features=None):
        if "feature_contributions" not in self.narrators:
            self.narrators["feature_contributions"] = explingo.Narrator(
                llm=self.llm,
                openai_api_key=self.openai_api_key,
                gpt_model_name=self.llm_model,
                explanation_format="(feature, feature_value, SHAP contribution)",
                context=self.context_description,
                labeled_train_data=self.training_examples.get("feature_contribution"),
            )

        narrator = self.narrators["feature_contributions"]

        if num_features is None:
            num_features = self.num_features

        narrative_explanations = []
        parsed_explanations = self.parse_feature_contribution_explanation_for_llm(
            explanation, num_features=num_features
        )

        for parsed_explanation in parsed_explanations:
            narrative_explanations.append(narrator.narrate(parsed_explanation))
        return NarrativeExplanation(narrative_explanations)

    @staticmethod
    def parse_feature_contribution_explanation_for_llm(explanation, num_features=None):
        explanations = explanation.get_top_features(num_features=num_features)
        parsed_explanations = []
        for explanation in explanations:
            strings = []
            for feature, value, contribution in zip(
                explanation[0].index, explanation[1], explanation[0]
            ):
                strings.append(f"({feature}, {value}, {contribution})")
            parsed_explanations.append(", ".join(strings))
        return parsed_explanations
