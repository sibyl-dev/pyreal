from pyreal.transformers import TransformerBase
from openai import OpenAI
from pyreal.explanation_types import NarrativeExplanation


class NarrativeTransformer(TransformerBase):
    def __init__(
        self,
        openai_client=None,
        openai_api_key=None,
        num_features=5,
        llm_model="gpt3.5",
        detail_level="high",
        context_description="",
        max_tokens=200,
        temperature=0.5,
        training_examples=None,
        **kwargs,
    ):
        """
        Transforms explanations to narrative (natural-language) form.
        Args:
            x_orig (DataFrame of shape (n_instances, n_features):
                Input to explain
            num_features (int):
                Number of features to include in the explanation, when relevant.
                If None, all features will be included
            llm_model (string):
                One of ["gpt3.5", "gpt4"]. LLM model to use to generate the explanation.
                GPT4 may provide better results, but is more expensive.
            detail_level (string):
                One of ["high", "low"]. Level of detail to include in the explanation.
                High detail should include precise contribution values. Low detail
                will include only basic information about features used.
            context_description (string):
                Description of the model's prediction task, in sentence format. This will be
                passed to the LLM and may help produce more accurate explanations.
                For example: "The model predicts the price of houses."
            max_tokens (int):
                Maximum number of tokens to use in the explanation
            temperature (float):
                LLM Temperature to use. Values closer to 1 will produce more creative values.
                Values closer to 0 will produce more consistent or conservative explanations.
            openai_client (OpenAI API client):
                OpenAI API client, with API key set. If None, the API key must be provided to the
                explainer at initialization.
            examples (dictionary of string:list of tuples):
                Few-shot training examples.
                Keys are explanation type (currently support: "feature_contributions")
                Values are lists of tuples, where the first element is the input to the model
                    and the second element is the example output.
                Use the RealApp train_llm functions to populate this
        Returns:
            list of strings of length n_instances
                Narrative version of feature contribution explanation, one item per instance
        """
        if openai_client is not None:
            self.openai_client = openai_client
        elif openai_api_key is not None:
            self.openai_client = OpenAI(api_key=openai_api_key)
        else:
            raise ValueError("Must provide openai_client or openai_api_key")

        self.num_features = num_features
        self.llm_model = llm_model
        self.detail_level = detail_level
        self.context_description = context_description
        self.max_tokens = max_tokens
        self.temperature = temperature

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

    def transform_explanation_feature_contribution(self, explanation, num_features=None):
        if num_features is None:
            num_features = self.num_features
        if self.llm_model == "gpt3.5":
            model = "gpt-3.5-turbo-0125"
        elif self.llm_model == "gpt4":
            model = "gpt-4-0125-preview"
        else:
            raise ValueError(
                "Invalid LLM model %s. Expected one of ['gpt3.5', 'gpt4']" % self.llm_model
            )
        if self.context_description is None:
            context_description = ""
        else:
            context_description = self.context_description.strip()
            if not context_description.endswith("."):
                context_description += "."
        prompt = (
            "You are helping users who do not have experience working with ML understand an ML"
            f" model's predictions. {context_description} I will give you feature contribution"
            " explanations, generated using SHAP, in (feature, feature_value, contribution )"
            " format. Convert the explanations into simple narratives. Do not use more tokens"
            " than necessary. Make your answers sound very natural, as if said in conversation. "
        )
        if self.detail_level == "low":
            prompt += (
                "Keep the explanations simple and easy to understand. Do not include exact"
                " contribution values. "
            )
        elif self.detail_level == "high":
            prompt += "Include all exact contribution values in your response. "
        else:
            raise ValueError(
                "Invalid detail_level %s. Expected one of ['high', 'low']" % self.detail_level
            )

        narrative_explanations = []
        base_messages = [{"role": "system", "content": prompt}]
        if self.training_examples.get("feature_contributions") is not None:
            for training_exp, training_narr in self.training_examples["feature_contributions"]:
                base_messages.append({"role": "user", "content": training_exp})
                base_messages.append({"role": "assistant", "content": training_narr})
        parsed_explanations = self.parse_feature_contribution_explanation_for_llm(
            explanation, num_features=num_features
        )
        for parsed_explanation in parsed_explanations:
            messages = base_messages + [{"role": "user", "content": parsed_explanation}]
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            narrative_explanations.append(response.choices[0].message.content)
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
