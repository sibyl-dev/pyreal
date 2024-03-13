import logging

from pyreal.explainers import (
    LocalFeatureContributionsBase,
    ShapFeatureContribution,
    SimpleCounterfactualContribution,
)

log = logging.getLogger(__name__)


def choose_algorithm():
    """
    Choose an algorithm based on the model type.
    Currently, shap is the only supported algorithm

    Returns:
        string (one of ["shap"])
            Explanation algorithm to use
    """
    return "shap"


class LocalFeatureContribution(LocalFeatureContributionsBase):
    """
    Generic LocalFeatureContribution wrapper

    A LocalFeatureContributions object wraps multiple local feature-based explanations. If no
    specific algorithm is requested, one will be chosen based on the information given.
    Currently, only SHAP is supported.

    Args:
        model (string filepath or model object):
           Filepath to the pickled model to explain, or model object with .predict() function
        x_train_orig (dataframe of shape (n_instances, x_orig_feature_count)):
           The training set for the explainer
        e_algorithm (string, one of ["shap", "simple"]):
           Explanation algorithm to use. If none, one will be chosen automatically based on model
           type
        shap_type (string, one of ["kernel", "linear"]):
            Type of shap algorithm to use, if e_algorithm="shap".
        **kwargs: see LocalFeatureContributionsBase args
    """

    def __init__(self, model, x_train_orig=None, e_algorithm=None, shap_type=None, **kwargs):
        if e_algorithm is None:
            e_algorithm = choose_algorithm()
        self.base_local_feature_contribution = None
        if e_algorithm == "shap":
            self.base_local_feature_contribution = ShapFeatureContribution(
                model, x_train_orig, shap_type=shap_type, **kwargs
            )
        elif e_algorithm == "simple":
            self.base_local_feature_contribution = SimpleCounterfactualContribution(
                model, x_train_orig, **kwargs
            )
        if self.base_local_feature_contribution is None:
            raise ValueError("Invalid algorithm type %s" % e_algorithm)

        super(LocalFeatureContribution, self).__init__(model, x_train_orig, **kwargs)

    def fit(self, x_train_orig=None, y_train=None):
        """
        Fit this explainer object

        Args:
            x_train_orig (DataFrame of shape (n_instances, n_features):
                Training set to fit on, required if not provided on initialization
            y_train:
                Targets of training set, required if not provided on initialization
        """
        self.base_local_feature_contribution.fit(x_train_orig, y_train)
        return self

    def produce_explanation(self, x_orig, **kwargs):
        """
        Gets the raw explanation.
        Args:
            x_orig (DataFrame of shape (n_instances, n_features):
                Input to explain

        Returns:
            FeatureContributionExplanation
                Contribution of each feature for each instance
        """
        return self.base_local_feature_contribution.produce_explanation(x_orig)

    def produce_narrative_explanation(
        self,
        x_orig,
        num_features=5,
        llm_model="gpt3.5",
        detail_level="high",
        context_description="",
        max_tokens=200,
        temperature=0.5,
    ):
        """
        Produces an explanation in narrative (natural-language) form.
        Args:
            x_orig (DataFrame of shape (n_instances, n_features):
                Input to explain
            num_features (int):
                Number of features to include in the explanation. If None, all features will be
                included
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

        Returns:
            list of strings of length n_instances
                Narrative version of feature contribution explanation, one item per instance
        """
        if self.openai_client is None:
            raise ValueError("OpenAI API key not set")
        return self.narrify(
            self.openai_client,
            self.base_local_feature_contribution.produce(x_orig),
            num_features=num_features,
            max_tokens=max_tokens,
            temperature=temperature,
            llm_model=llm_model,
            detail_level=detail_level,
            context_description=context_description,
        )

    def train_llm(
        self, x_train=None, live=True, provide_examples=False, num_inputs=5, num_features=3
    ):
        """
        Run the training process for the LLM model used to generate narrative feature
        contribution explanations.

        Args:
            x_train (DataFrame of shape (n_instances, n_features)):
                Training set to take sample inputs from. If None, the training set must be provided
                to the explainer at initialization.
            live (Boolean):
                If True, run the training process through CLI input/outputs. If False,
                this function will generate a shell training file that will need to be filled out
                and added to the RealApp manually.
            provide_examples (Boolean):
                If True, generate a base example of explanations at each step. This may make
                the process faster, but will incur costs to your OpenAI API account.
            num_inputs (int):
                Number of inputs to request.
            num_features (int):
                Number of features to include per explanation. If None, all features will be
                included

        Returns:
            None
        """
        if not live:
            raise NotImplementedError("Non-interactive training not yet implemented")
        if provide_examples and self.openai_client is None:
            raise ValueError(
                "OpenAI API key or client must be provided to provide examples during training"
            )
        x_train = self._get_x_train_orig(x_train)
        explanation = self.produce(x_train).get_top_features(num_features=num_features)
        narratives = []
        if num_inputs >= len(explanation):
            print("Warning: not enough inputs provided for training. Using all inputs")
            num_inputs = len(explanation)
        print("For each of the following inputs, please provide an appropriate narrative version.")
        for i in range(num_inputs):
            parsed_explanation = parse_explanation_for_llm(explanation[i])
            print(f"Input {i+1}: {parsed_explanation}")
            if provide_examples:
                print(self.narrify(explanation[i]))
            narratives.append(input("Narrative explanation:"))
        print(f"Training complete. Training data: {narratives}")

    @staticmethod
    def narrify(
        openai_client,
        explanation,
        num_features=None,
        llm_model="gpt3.5",
        detail_level="high",
        context_description="",
        max_tokens=200,
        temperature=0.5,
    ):
        """
        Generate a narrative explanation from a feature contribution explanation
        Args:
            openai_client (OpenAI API client):
                OpenAI API client, with API key set
            explanation (LocalFeatureContributionExplanation):
                Feature contribution explanations. Each row represents an instance, and each
                column a feature.
            num_features (int):
                Number of features to include in the explanation. If None, all features will be
                included
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

        Returns:
            DataFrame of shape (n_instances, n_features)
                Narrative explanation
        """
        if llm_model == "gpt3.5":
            model = "gpt-3.5-turbo-0125"
        elif llm_model == "gpt4":
            model = "gpt-4-0125-preview"
        else:
            raise ValueError(
                "Invalid LLM model %s. Expected one of ['gpt3.5', 'gpt4']" % llm_model
            )
        if context_description is None:
            context_description = ""
        if context_description:
            context_description = context_description.strip()
            if not context_description.endswith("."):
                context_description += "."
        prompt = (
            "You are helping users who do not have experience working with ML understand an ML"
            f" model's predictions. {context_description} I will give you feature contribution"
            " explanations, generated using SHAP, in (feature, feature_value, contribution )"
            " format. Convert the explanations into simple narratives. Do not use more tokens"
            " than necessary. Make your answers sound very natural, as if said in conversation. "
        )
        if detail_level == "low":
            prompt += (
                "Keep the explanations simple and easy to understand. Do not include exact"
                " contribution values. "
            )
        elif detail_level == "high":
            prompt += "Include all exact contribution values in your response. "
        else:
            raise ValueError(
                "Invalid detail_level %s. Expected one of ['high', 'low']" % detail_level
            )
        explanation = explanation.get_top_features(num_features=num_features)
        narrative_explanations = []
        for row in explanation:
            parsed_explanation = parse_explanation_for_llm(row)
            response = openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": parsed_explanation},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            narrative_explanations.append(response.choices[0].message.content)
        return narrative_explanations


def parse_explanation_for_llm(explanation):
    strings = []
    for feature, value, contribution in zip(explanation[0].index, explanation[1], explanation[0]):
        strings.append(f"({feature}, {value}, {contribution})")
    return ", ".join(strings)
