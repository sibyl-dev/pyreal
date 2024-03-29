from abc import ABC

import numpy as np

from pyreal.explainers import ExplainerBase


class LocalFeatureContributionsBase(ExplainerBase, ABC):
    """
    Base class for LocalFeatureContributions explainer objects. Abstract class

    A LocalFeatureContributionsBase object explains a machine learning prediction by assigning an
    importance or contribution score to every feature. LocalFeatureContributionBase objects explain
    by taking an instance and returning one number per feature, per instance.

    Args:
        model (string filepath or model object):
           Filepath to the pickled model to explain, or model object with .predict() function
        x_train_orig (dataframe of shape (n_instances, x_orig_feature_count)):
           The training set for the explainer
        **kwargs: see base Explainer args
    """

    def __init__(self, model, x_train_orig=None, **kwargs):
        self.llm_training_data = None
        super(LocalFeatureContributionsBase, self).__init__(model, x_train_orig, **kwargs)

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
        if explanations is None:
            explanations = []
            for i in range(n_iterations - 1):
                if with_fit:
                    self.fit()
                explanations.append(
                    self.produce(self.x_train_orig_subset.iloc[0:n_rows]).get().to_numpy()
                )

        return np.max(np.var(explanations, axis=0))

    def produce_narrative_explanation(
        self,
        x_orig,
        num_features=5,
        llm_model="gpt3.5",
        detail_level="high",
        context_description="",
        max_tokens=200,
        temperature=0.5,
        openai_client=None,
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
            openai_client (OpenAI API client):
                OpenAI API client, with API key set. If None, the API key must be provided to the
                explainer at initialization.

        Returns:
            list of strings of length n_instances
                Narrative version of feature contribution explanation, one item per instance
        """
        if openai_client is None:
            if self.openai_client is None:
                raise ValueError("OpenAI API key or client must be provided to produce narrative")
            openai_client = self.openai_client

        return self.narrify(
            openai_client,
            self.produce(x_orig),
            num_features=num_features,
            max_tokens=max_tokens,
            temperature=temperature,
            llm_model=llm_model,
            detail_level=detail_level,
            context_description=context_description,
            few_shot_training_examples=self.llm_training_data,
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
                and added to the RealApp manually. Currently only live training is supported.
            provide_examples (Boolean):
                If True, generate a base example of explanations at each step. This may make
                the process faster, but will incur costs to your OpenAI API account.
            num_inputs (int):
                Number of inputs to request.
            num_features (int):
                Number of features to include per explanation. If None, all features will be
                included

        Returns:
            list of (explanation, narrative) pairs
                The generated training data
        """
        if not live:
            raise NotImplementedError("Non-interactive training not yet implemented")
        if provide_examples and self.openai_client is None:
            raise ValueError(
                "OpenAI API key or client must be provided to provide examples during training"
            )
        x_train = self._get_x_train_orig(x_train)
        if num_inputs > len(x_train):
            print(
                "Warning: number of inputs in x_train is less than num_inputs, using all available"
                " inputs"
            )
        else:
            x_train = x_train.sample(num_inputs)
        explanation = self.produce(x_train)
        parsed_explanations = parse_explanation_for_llm(explanation, num_features=num_features)
        narratives = []
        print("For each of the following inputs, please provide an appropriate narrative version.")
        for i in range(num_inputs):
            instruction = ""
            parsed_explanation_formatted = parsed_explanations[i].replace("), ", "),\n")
            instruction += (
                f"Input {i+1} (feature, value, contribution):\n{parsed_explanation_formatted}\n"
            )
            if provide_examples:
                example = LocalFeatureContributionsBase.narrify(
                    self.openai_client, explanation[i], num_features=num_features
                )[0]
                instruction += f"Example: {example}\n"
                instruction += "Narrative explanation ('k' to keep example, 'q' to quit): "
                narrative = input(instruction)
                if narrative.lower() == "k":
                    narrative = example
            else:
                instruction += "Narrative explanation ('q' to quit): "
                narrative = input(instruction)
            if narrative.lower() == "q":
                break
            narratives.append((parsed_explanations[i], narrative))
        if len(narratives) > 0 and input("Save training data? (y/n): ").lower() == "y":
            self.llm_training_data = narratives
            print(f"Training complete. Training data: {narratives}")
        else:
            print("Training data not saved.")
        return narratives

    def clear_llm_training_data(self):
        """
        Remove few-shot training examples from the explainer
        """
        self.llm_training_data = None

    def set_llm_training_data(self, training_data):
        """
        Manually set llm training data

        Args:
            training_data (list of (explanation, narrative) pairs):
                Training examples to use for few-shot learning. If provided, the LLM will be
                trained on these examples before generating the explanation.
        """
        self.llm_training_data = training_data

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
        few_shot_training_examples=None,
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
            few_shot_training_examples (list of (explanation, narrative) pairs):
                Training examples to use for few-shot learning. If provided, the LLM will be
                trained on these examples before generating the explanation.

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
        # explanation = explanation.get_top_features(num_features=num_features)
        narrative_explanations = []
        base_messages = [{"role": "system", "content": prompt}]
        if few_shot_training_examples is not None:
            for training_exp, training_narr in few_shot_training_examples:
                base_messages.append({"role": "user", "content": training_exp})
                base_messages.append({"role": "assistant", "content": training_narr})
        parsed_explanations = parse_explanation_for_llm(explanation, num_features=num_features)
        for parsed_explanation in parsed_explanations:
            messages = base_messages + [{"role": "user", "content": parsed_explanation}]
            response = openai_client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            narrative_explanations.append(response.choices[0].message.content)
        return narrative_explanations


def parse_explanation_for_llm(explanation, num_features=None):
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
