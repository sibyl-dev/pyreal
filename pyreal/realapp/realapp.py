import pandas as pd
from openai import OpenAI

from pyreal.explainers import (
    Explainer,
    GlobalFeatureImportance,
    LocalFeatureContribution,
    SimilarExamples,
)
from pyreal.explanation_types import NarrativeExplanation
from pyreal.transformers import (
    NarrativeTransformer,
    run_transformers,
    sklearn_pipeline_to_pyreal_transformers,
)
from pyreal.utils import get_top_contributors


def format_feature_contribution_output(
    explanation, ids=None, series=False, optimized=False, include_average_values=False
):
    """
    Format Pyreal FeatureContributionExplanation objects into Local Feature Contribution outputs
    Args:
        explanation (FeatureContributionExplanation):
            Pyreal Explanation object to parse
        ids (list of strings or ints):
            List of row ids
        series (Boolean):
            If True, the produce function was passed a series input
        optimized (Boolean):
            If True, return in a simple DataFrame format
        include_average_values (Boolean):
            If True, include the expected (average) value of each feature in the output
    Returns:
        DataFrame (if series), else {"id" -> DataFrame}
            One dataframe per id, with each row representing a feature, and four columns:
                Feature Name    Feature Value   Contribution    Average/Mode
        if optimized: DataFrame, with one row per instance and one column per feature
    """
    if ids is None:
        ids = explanation.get().index
    if optimized:
        return explanation.get().set_index(ids), explanation.get_values().set_index(ids)
    explanation_dict = {}
    for i, row_id in enumerate(ids):
        contributions = explanation.get().iloc[i, :]
        values = explanation.get_values().iloc[i, :].loc[contributions.index]
        feature_names = contributions.index

        if include_average_values:
            if explanation.get_average_values() is None:
                raise ValueError(
                    "Requested average values to be included in explanation, but explainer did not"
                    " provide them"
                )
            average_mode = explanation.get_average_values()[contributions.index]
            explanation_dict[row_id] = pd.DataFrame.from_dict(
                {
                    "Feature Name": feature_names.values,
                    "Feature Value": values.values,
                    "Contribution": contributions.values,
                    "Average/Mode": average_mode.values,
                }
            )
        else:
            explanation_dict[row_id] = pd.DataFrame.from_dict(
                {
                    "Feature Name": feature_names.values,
                    "Feature Value": values.values,
                    "Contribution": contributions.values,
                }
            )

    if series:
        return explanation_dict[next(iter(explanation_dict))]
    return explanation_dict


def format_feature_importance_output(explanation, optimized=False):
    """
    Format Pyreal FeatureImportanceExplanation objects into Global Feature Importance outputs
    Args:
        explanation (FeatureImportanceExplanation):
            Pyreal Explanation object to parse
        optimized (Boolean):
            If True, return in a simple DataFrame format

    Returns:
        DataFrame with a Feature Name column and an Importance column (if not optimized),
        else a single row DataFrame with one column per feature
    """
    importances = explanation.get()
    if optimized:
        return importances
    return pd.DataFrame(
        {"Feature Name": importances.columns, "Importance": importances.squeeze()}
    ).reset_index(drop=True)


def format_similar_examples_output(
    explanation,
    ids=None,
    series=False,
    y_format_func=None,
    optimized=False,
    id_column_name=None,
    training_ids=None,
):
    """
    Format Pyreal SimilarExamples objects into Similar Examples outputs
    Args:
        explanation (SimilarExampleExplanation):
            Pyreal Explanation object to parse
        ids (list of strings or ints):
            List of row ids
        series (Boolean):
            If True, the produce function was passed a series input
        y_format_func (function):
            Function to use to format ground truth values
                optimized (Boolean)
        optimized (Boolean):
            Current a no-op, included for consistency
        id_column (string or int):
            Name of column that contains item ids in input data
        training_ids (Series):
            Series of ids associated with X_train

    Returns:
        {"X": DataFrame, "y": Series, "Input": Series} (if series),
                else {"id" -> {"X": DataFrame, "y": Series, "Input": Series}}
            X is the examples, ordered from top to bottom by similarity to input and
            y is the corresponding y values
            Input is the original input in the same feature space
    """
    result = {}
    if ids is None:
        ids = explanation.get_row_ids()
    for key, id_column in enumerate(ids):
        examples = explanation.get_examples(row_id=key)
        targets = explanation.get_targets(row_id=key)
        if y_format_func is not None:
            targets = targets.apply(y_format_func)
        result[id_column] = {
            "X": examples,
            "y": targets,
            "Input": explanation.get_values().iloc[key, :],
        }
        if training_ids is not None:
            if id_column is None:
                id_column = "row_id"
            result[id_column]["X"][id_column_name] = training_ids.loc[result[id_column]["X"].index]
    if series:
        return result[next(iter(result))]
    return result


def format_narratives(narratives, ids, series=False, optimized=False):
    if optimized or series:
        return narratives
    return {row_id: narr for row_id, narr in zip(ids, narratives)}


class RealApp:
    """
    Maintains all information about a Pyreal application to generate explanations
    """

    def __init__(
        self,
        models,
        X_train_orig=None,
        y_train=None,
        transformers=None,
        feature_descriptions=None,
        active_model_id=None,
        classes=None,
        class_descriptions=None,
        pred_format_func=None,
        fit_transformers=False,
        id_column=None,
        openai_api_key=None,
        llm=None,
        context_description="",
    ):
        """
        Initialize a RealApp object

        Args:
            models (model object, list of models, or dict of model_id:model):
                Model(s) for this application
            X_train_orig (DataFrame of shape (n_instances,n_features):
                Training data for models. If None, must be provided when preparing explainers.
            y_train (DataFrame of shape (n_instances,)):
                The y values for the dataset
            transformers (Transformer object or list of Transformer objects):
                Transformers for this application
            feature_descriptions (dictionary of feature_name:feature_description):
                Mapping of default feature names to readable names
            active_model_id (string or int):
                ID of model to store as active model, if None, this is set to the first model
            classes (array):
                List of class names returned by the model, in the order that the internal model
                considers them if applicable.
                Can be automatically extracted if model is an sklearn classifier
                None if model is not a classifier
            class_descriptions (dict):
                Interpretable descriptions of each class
                None if model is not a classifier
            pred_format_func (function):
                Function to format model prediction outputs
            fit_transformers (Boolean):
                If True, fit the transformers to X_train_orig on initialization
            id_column (string or int):
                Name of column that contains item ids in input data
            openai_api_key (string):
                OpenAI API key. Required for GPT narrative explanations, unless openai client
                is provided
            llm (LLM model object):
                Local LLM object or LLM client object to use to generate narratives.
            context_description (string):
                Description of the model's prediction task, in sentence format. This is used by
                LLM model for narrative explanations.
                For example: "The model predicts the price of houses."
        """
        self.expect_model_id = False
        if isinstance(models, dict):
            self.expect_model_id = True
            self.models = models
        elif isinstance(models, list):
            self.models = {i: models[i] for i in range(0, len(models))}
        else:  # assume single model given
            self.models = {0: models}

        if active_model_id is not None:
            if active_model_id not in self.models:
                raise ValueError("active_model_id not in models")
            self.active_model_id = active_model_id
        else:
            self.active_model_id = next(iter(self.models))

        self.id_column = id_column

        self.training_ids_for_se = None
        if (
            X_train_orig is not None
            and self.id_column is not None
            and self.id_column in X_train_orig
        ):
            self.training_ids_for_se = X_train_orig[self.id_column]
            self.X_train_orig = X_train_orig.drop(columns=self.id_column)
        else:
            self.X_train_orig = X_train_orig
        self.y_train = y_train

        self.classes = classes
        self.class_descriptions = class_descriptions
        self.pred_format_func = pred_format_func

        if transformers is None or isinstance(transformers, list):
            self.transformers = transformers
        else:  # assume single transformer given
            self.transformers = [transformers]
        self.feature_descriptions = feature_descriptions

        self.llm = llm
        self.openai_api_key = openai_api_key

        if fit_transformers:
            # Hacky way of fitting transformers, may want to clean up later
            Explainer(
                self.models[next(iter(self.models))],
                self.X_train_orig,
                transformers=self.transformers,
                fit_transformers=True,
            )

        # Base explainer used for general transformations and model predictions
        # Also validates data, model, and transformers
        self.base_explainers = {
            model_id: self._make_base_explainer(self.models[model_id]) for model_id in self.models
        }

        self.explainers = {}  # Dictionary of dictionaries:
        # {"explanation_type": {"algorithm":Explainer} }

        if context_description is None:
            context_description = ""
        self.context_description = context_description

    def _make_base_explainer(self, model):
        """
        Make a base explainer for model.

        Args:
            model (model object):
                The model to be explained by this explainer
        Returns:
            Explainer
                The explainer
        """
        return Explainer(
            model,
            transformers=self.transformers,
            feature_descriptions=self.feature_descriptions,
        )

    def _explainer_exists(self, explanation_type, algorithm):
        """
        Check if the requested explainer exists

        Args:
            explanation_type (string):
                Code for explanation_type
            algorithm (string):
                Name of algorithm

        Returns:
            Boolean
                True if the specified explainer exists, False otherwise
        """
        if explanation_type in self.explainers:
            if algorithm in self.explainers[explanation_type]:
                return True
        return False

    def _add_explainer(self, explanation_type, algorithm, explainer):
        """
        Add the specified explainer to this RealApp

        Args:
            explanation_type (string):
                Code for explanation_type
            algorithm (string):
                Name of algorithm
            explainer (Explainer):
                Explainer to add
        """
        if explanation_type not in self.explainers:
            self.explainers[explanation_type] = {}
        self.explainers[explanation_type][algorithm] = explainer

    def _get_explainer(self, explanation_type, algorithm=None):
        """
        Get the requested explainer

        Args:
            explanation_type (string):
                Code for explanation_type
            algorithm (string):
                Name of algorithm. If None, return all valid explainer of the requested type.

        Returns:
            Explainer or False
                The requested explainer, of False if not yet fitted
        """
        if explanation_type not in self.explainers:
            return False
        if algorithm is None:
            return self.explainers[explanation_type]
        if algorithm not in self.explainers[explanation_type]:
            return False
        return self.explainers[explanation_type][algorithm]

    def _produce_explanation_helper(
        self,
        explanation_type_code,
        algorithm,
        prepare_explainer_func,
        format_output_func,
        format_output=True,
        x_train_orig=None,
        y_train=None,
        x_orig=None,
        model_id=None,
        force_refit=False,
        training_size=None,
        prepare_kwargs=None,
        produce_kwargs=None,
        format_kwargs=None,
        narrative=False,
    ):
        """
        Produce an explanation from a specified Explainer

        Args:
            explanation_type_code (string):
                Code for explanation_type
            algorithm (string):
                Name of algorithm
            prepare_explainer_func (function):
                Function that initializes and fits the appropriate explainer
            format_output_func (function):
                Function that formats Explanation objects into the appropriate output format
            format_output (Boolean):
                If False, return output in simple format. Formatted outputs are more usable
                but take longer to generate.
            x_train_orig (DataFrame of shape (n_instances, n_features)):
                Training data, if not provided at initialization.
            y_train (DataFrame or Series):
                Training targets, if not provided at initialization
            x_orig (DataFrame):
                Data to explain, required for local explanations
            model_id (string or int):
                ID of model to explain
            force_refit (Boolean):
                If True, initialize and fit a new explainer even if the appropriate explainer
                already exists
            prepare_kwargs (dict):
                Additional parameters for explainer init function
            produce_kwargs (dict):
                Additional parameters for explainer produce function
            format_kwargs (dict):
                Additional parameters for format function
            narrative (Boolean):
                If True, use explainer's produce_narrative_explanation() function

        Returns:
            Type varies by explanation type
                The explanation
        """
        if model_id is None:
            model_id = self.active_model_id

        if prepare_kwargs is None:
            prepare_kwargs = {}
        if produce_kwargs is None:
            produce_kwargs = {}
        if format_kwargs is None:
            format_kwargs = {}

        if self._explainer_exists(explanation_type_code, algorithm) and not force_refit:
            explainer = self._get_explainer(explanation_type_code, algorithm)
        else:
            explainer = prepare_explainer_func(
                model_id=model_id,
                algorithm=algorithm,
                x_train_orig=x_train_orig,
                y_train=y_train,
                training_size=training_size,
                **prepare_kwargs
            )

        if narrative and not hasattr(explainer, "produce_narrative_explanation"):
            raise ValueError("narrative explanations not supported for this explainer")

        if x_orig is not None:
            series = x_orig.ndim == 1
            ids = None

            if self.id_column is not None and self.id_column in x_orig:
                ids = x_orig[self.id_column]
                if series:  # If x was a series, ids will now be a scaler
                    ids = [ids]
                x_orig = x_orig.drop(self.id_column, axis=x_orig.ndim - 1)

            if narrative:
                narratives = explainer.produce_narrative_explanation(x_orig, **produce_kwargs)
                if ids is None:
                    ids = x_orig.index
                return format_narratives(
                    narratives.get(),
                    ids=ids,
                    series=series,
                    optimized=not format_output,
                )
            else:
                explanation = explainer.produce(x_orig, **produce_kwargs)
                if isinstance(explanation, NarrativeExplanation):
                    return format_narratives(
                        explanation.get(),
                        ids=ids,
                        series=series,
                        optimized=not format_output,
                    )
                else:
                    return format_output_func(
                        explanation,
                        ids,
                        optimized=not format_output,
                        series=series,
                        **format_kwargs
                    )
        else:
            if narrative:
                return explainer.produce_narrative_explanation(**produce_kwargs).get()
            else:
                explanation = explainer.produce(**produce_kwargs)
                return format_output_func(
                    explanation, optimized=not format_output, **format_kwargs
                )

    def add_model(self, model, model_id=None):
        """
        Add a model

        Args:
            model (model object):
                Model to add
            model_id (string or int):
                ID of model. Must be provided when models was originally given as a dictionary. If
                none, model ID will be incremented from previous model
        """
        if model_id is None:
            if self.expect_model_id is True:
                raise ValueError(
                    "Models was originally provided as a dictionary, so you must provide a"
                    " model_id when adding a model"
                )
            else:
                model_id = len(self.models) + 1
        self.models[model_id] = model

    def set_active_model_id(self, active_model_id):
        """
        Set a new active model

        Args:
            active_model_id (int or string):
                New model id to set as active model
        """
        if active_model_id not in self.models:
            raise ValueError("active_model_id not in models")
        self.active_model_id = active_model_id

    def get_active_model(self):
        """
        Return the active model

        Returns:
            (model object)
                The active model
        """
        return self.models[self.active_model_id]

    def predict(self, x, model_id=None, as_dict=None, format=True):
        """
        Predict on x using the active model or model specified by model_id

        Args:
            x (DataFrame of shape (n_instances, n_features) or Series of len n_features):
                Data to predict on
            model_id (int or string):
                Model to use for prediction
            as_dict (Boolean):
                If False, return predictions as a single Series/List. Otherwise, return
                in {row_id: pred} format. Defaults to True if x is a DataFrame, False otherwise
            format (Boolean):
                If False, do not run the realapp's format function on this output

        Returns:
            (model return type)
                Model prediction on x
        """
        if as_dict is None:
            as_dict = x.ndim > 1
        if self.id_column is not None and self.id_column in x:
            ids = x[self.id_column]
            x = x.drop(self.id_column, axis=x.ndim - 1)
        else:
            ids = x.index
        if model_id is None:
            model_id = self.active_model_id

        preds = self.base_explainers[model_id].model_predict(x)
        if not as_dict:
            if format and self.pred_format_func is not None:
                return [self.pred_format_func(pred) for pred in preds]
            return preds
        preds_dict = {}
        for i, row_id in enumerate(ids):
            if format and self.pred_format_func is not None:
                preds_dict[row_id] = self.pred_format_func(preds[i])
            else:
                preds_dict[row_id] = preds[i]
        return preds_dict

    def predict_proba(self, x, model_id=None, as_dict=None, format=True):
        """
        Return the predicted probabilities of x using the active model or
        model specified by model_id, only if the model has a predict_proba method

        Args:
            x (DataFrame of shape (n_instances, n_features) or Series of len n_features):
                Data to predict on
            model_id (int or string):
                Model to use for prediction
            as_dict (Boolean):
                If False, return predictions as a single Series/List. Otherwise, return
                in {row_id: pred} format. Defaults to True if x is a DataFrame, False otherwise
            format (Boolean):
                If False, do not run the realapp's format function on this output

        Returns:
            (model return type)
                Model prediction on x in terms of probability
        """
        if as_dict is None:
            as_dict = x.ndim > 1
        if self.id_column is not None and self.id_column in x:
            ids = x[self.id_column]
            x = x.drop(self.id_column, axis=x.ndim - 1)
        else:
            ids = x.index
        if model_id is None:
            model_id = self.active_model_id

        preds = self.base_explainers[model_id].model_predict_proba(x)
        if not as_dict:
            if format and self.pred_format_func is not None:
                return [self.pred_format_func(pred) for pred in preds]
            return preds
        preds_dict = {}
        for i, row_id in enumerate(ids):
            if format and self.pred_format_func is not None:
                preds_dict[row_id] = self.pred_format_func(preds[i])
            else:
                preds_dict[row_id] = preds[i]
        return preds_dict

    def prepare_feature_contributions(
        self,
        x_train_orig=None,
        y_train=None,
        model_id=None,
        algorithm=None,
        shap_type=None,
        training_size=None,
    ):
        """
        Initialize and fit a local feature contribution explainer

        Args:
            model_id (int or string):
                Model id to explain
            x_train_orig (DataFrame of shape (n_instances, n_features)):
                Training data, if not provided at initialization.
            y_train (DataFrame or Series):
                Training targets, if not provided at initialization
            algorithm (string):
                LFC algorithm to use
            shap_type (string):
                If algorithm is "shap", type of shap to use
            training_size (int):
                Number of rows to use in fitting explainer

        Returns:
            A fit LocalFeatureContribution explainer
        """
        if algorithm is None:
            algorithm = "shap"

        if model_id is None:
            model_id = self.active_model_id

        explainer = LocalFeatureContribution(
            self.models[model_id],
            transformers=self.transformers,
            feature_descriptions=self.feature_descriptions,
            e_algorithm=algorithm,
            shap_type=shap_type,
            classes=self.classes,
            class_descriptions=self.class_descriptions,
            training_size=training_size,
            llm=self.llm,
            openai_api_key=self.openai_api_key,
        )
        explainer.fit(self._get_x_train_orig(x_train_orig), self._get_y_train(y_train))
        self._add_explainer("lfc", algorithm, explainer)
        return explainer

    def produce_feature_contributions(
        self,
        x_orig,
        model_id=None,
        x_train_orig=None,
        y_train=None,
        algorithm=None,
        format_output=True,
        shap_type=None,
        force_refit=False,
        training_size=None,
        num_features=None,
        select_by="absolute",
        include_average_values=False,
    ):
        """
        Produce a feature contribution explanation

        Args:
            x_orig (DataFrame of shape (n_instances, n_features) or Series of length (n_features)):
                Input(s) to explain
            model_id (string or int):
                ID of model to explain
            x_train_orig (DataFrame):
                Data to fit on, if not provided during initialization
            y_train (DataFrame or Series):
                Training targets to fit on, if not provided during initialization
            algorithm (string):
                Name of algorithm
            format_output (Boolean):
                If False, return output as a single DataFrame. Formatted outputs are more usable
                but take longer to generate.
            shap_type (string):
                If algorithm="shap", type of SHAP explainer to use
            force_refit (Boolean):
                If True, initialize and fit a new explainer even if the appropriate explainer
                already exists
            training_size (int):
                Number of rows to use in fitting explainer
            num_features (int):
                Number of features to include in the explanation. If None, include all features
            select_by (one of "absolute", "min", "max"):
                If `num_features` is not None, method to use for selecting which features to show.
                Not used if num_features is None
            include_average_values (Boolean):
                If True, include the average/mode value of each feature in the output

        Returns:
            dictionary (if x_orig is DataFrame) or DataFrame (if x_orig is Series)
                One dataframe per id, with each row representing a feature, and four columns:
                Feature Name    Feature Value   Contribution    Average/Mode
        """
        if algorithm is None:
            algorithm = "shap"

        exp = self._produce_explanation_helper(
            "lfc",
            algorithm,
            self.prepare_feature_contributions,
            format_feature_contribution_output,
            x_train_orig=x_train_orig,
            y_train=y_train,
            x_orig=x_orig,
            format_output=format_output,
            model_id=model_id,
            force_refit=force_refit,
            training_size=training_size,
            prepare_kwargs={"shap_type": shap_type},
            format_kwargs={"include_average_values": include_average_values},
        )
        if num_features is not None:
            return {
                row_id: get_top_contributors(
                    exp[row_id], num_features=num_features, select_by=select_by
                )
                for row_id in exp
            }
        else:
            return exp

    def produce_narrative_feature_contributions(
        self,
        x_orig,
        model_id=None,
        x_train_orig=None,
        y_train=None,
        algorithm=None,
        shap_type=None,
        force_refit=False,
        training_size=None,
        format_output=True,
        num_features=5,
        select_by="absolute",
        gpt_model_type="gpt-3.5",
        context_description=None,
        max_tokens=200,
    ):
        """
        Produce a feature contribution explanation, formatted in natural language sentence
        format using LLMs.
        Do not use this function if your transformer list ends with a NarrativeTransformer -
        simply call produce_feature_contributions instead.

        Args:
            x_orig (DataFrame of shape (n_instances, n_features) or Series of length (n_features)):
                Input(s) to explain
            model_id (string or int):
                ID of model to explain
            x_train_orig (DataFrame):
                Data to fit on, if not provided during initialization
            y_train (DataFrame or Series):
                Training targets to fit on, if not provided during initialization
            algorithm (string):
                Name of algorithm
            shap_type (string):
                If algorithm="shap", type of SHAP explainer to use
            force_refit (Boolean):
                If True, initialize and fit a new explainer even if the appropriate explainer
                already exists
            training_size (int):
                Number of rows to use in fitting explainer
            format_output (bool):
                If False, return output as a single list of narratives. Formatted outputs are more
                usable, but formatting may slow down runtimes on larger inputs
            num_features (int):
                Number of features to include in the explanation. If None, include all features
            select_by (one of "absolute", "min", "max"):
                If `num_features` is not None, method to use for selecting which features to show.
                Not used if num_features is None
            gpt_model_type (string):
                One of ["gpt3.5", "gpt4"]. LLM model to use to generate the explanation.
                GPT4 may provide better results, but is more expensive.
            context_description (string):
                Description of the model's prediction task, in sentence format. This will be
                passed to the LLM and may help produce more accurate explanations.
                For example: "The model predicts the price of houses."
            max_tokens (int):
                Maximum number of tokens to use in the explanation

        Returns:
            dictionary (if x_orig is DataFrame) or DataFrame (if x_orig is Series)
                One dataframe per id, with each row representing a feature, and four columns:
                Feature Name    Feature Value   Contribution    Average/Mode
        """
        for transformer in self.transformers:
            if isinstance(transformer, NarrativeTransformer):
                raise ValueError(
                    "Currently we do not support using produce_narrative functions when"
                    " NarrativeTransformers are passed in. Either remove the NarrativeTransformer"
                    " and call this function,or simply call produce_feature_contributions using"
                    " the NarrativeTransformer"
                )

        if algorithm is None:
            algorithm = "shap"

        if context_description is None:
            context_description = self.context_description

        exp = self._produce_explanation_helper(
            "lfc",
            algorithm,
            self.prepare_feature_contributions,
            format_feature_contribution_output,
            x_train_orig=x_train_orig,
            y_train=y_train,
            x_orig=x_orig,
            model_id=model_id,
            force_refit=force_refit,
            training_size=training_size,
            format_output=format_output,
            prepare_kwargs={"shap_type": shap_type},
            produce_kwargs={
                "gpt_model_type": gpt_model_type,
                "max_tokens": max_tokens,
                "num_features": num_features,
                "context_description": context_description,
            },
            narrative=True,
        )
        return exp

    def prepare_feature_importance(
        self,
        x_train_orig=None,
        y_train=None,
        model_id=None,
        algorithm=None,
        shap_type=None,
        training_size=None,
    ):
        """
        Initialize and fit a global feature importance explainer

        Args:
            model_id (int or string):
                Model id to explain
            x_train_orig (DataFrame of shape (n_instances, n_features)):
                Training data, if not provided at initialization.
            y_train (DataFrame or Series):
                Training targets, if not provided at initialization
            algorithm (string):
                GFI algorithm to use
            shap_type (string):
                If algorithm is "shap", type of shap to use
            training_size (int):
                Number of rows to use in fitting explainer

        Returns:
            A fit GlobalFeatureImportance explainer
        """
        if algorithm is None:
            algorithm = "shap"

        if model_id is None:
            model_id = self.active_model_id

        explainer = GlobalFeatureImportance(
            self.models[model_id],
            transformers=self.transformers,
            feature_descriptions=self.feature_descriptions,
            e_algorithm=algorithm,
            classes=self.classes,
            class_descriptions=self.class_descriptions,
            shap_type=shap_type,
            training_size=training_size,
        )
        explainer.fit(self._get_x_train_orig(x_train_orig), self._get_y_train(y_train))
        self._add_explainer("gfi", algorithm, explainer)
        return explainer

    def produce_feature_importance(
        self,
        model_id=None,
        x_train_orig=None,
        y_train=None,
        algorithm=None,
        format_output=True,
        shap_type=None,
        force_refit=False,
        training_size=None,
        num_features=None,
        select_by="absolute",
    ):
        """
        Produce a GlobalFeatureImportance explainer

        Args:
            model_id (string or int):
                ID of model to explain
            x_train_orig (DataFrame):
                Data to fit on, if not provided during initialization
            y_train (DataFrame or Series):
                Training targets to fit on, if not provided during initialization
            algorithm (string):
                Name of algorithm
            format_output (Boolean):
                If False, return output as a single DataFrame. Formatted outputs are more usable
                but take longer to generate.
            shap_type (string):
                If algorithm="shap", type of SHAP explainer to use
            force_refit (Boolean):
                If True, initialize and fit a new explainer even if the appropriate explainer
                already exists
            training_size (int):
                Number of rows to use in fitting explainer
            num_features (int):
                Number of features to include in the explanation. If None, include all features
            select_by (one of "absolute", "min", "max"):
                If `num_features` is not None, method to use for selecting which features to show.
                Not used if num_features is None

        Returns:
            DataFrame with a Feature Name column and an Importance column
        """
        if algorithm is None:
            algorithm = "shap"

        exp = self._produce_explanation_helper(
            "gfi",
            algorithm,
            self.prepare_feature_importance,
            format_feature_importance_output,
            model_id=model_id,
            x_train_orig=x_train_orig,
            y_train=y_train,
            format_output=format_output,
            force_refit=force_refit,
            training_size=training_size,
            prepare_kwargs={"shap_type": shap_type},
        )
        if num_features is not None:
            return get_top_contributors(exp, num_features=num_features, select_by=select_by)
        else:
            return exp

    def prepare_similar_examples(
        self,
        x_train_orig=None,
        y_train=None,
        model_id=None,
        algorithm=None,
        training_size=None,
        standardize=False,
        fast=True,
    ):
        """
        Initialize and fit a nearest neighbor explainer

        Args:
            model_id (int or string):
                Model id to explain
            x_train_orig (DataFrame of shape (n_instances, n_features)):
                Training data, if not provided at initialization.
            y_train (DataFrame or Series):
                Training targets, if not provided at initialization
            algorithm (string):
                NN algorithm to use (current options: [nn])
            training_size (int):
                Number of rows to use in fitting explainer
            standardize (Boolean):
                If True, standardize data before using it to get similar examples.
                Recommended if model-ready data is not already standardized
            fast (Boolean):
                If True, use a faster algorithm for getting similar examples (disable if faiss
                dependency not available)

        Returns:
            A fit SimilarExamples explainer
        """
        if algorithm is None:
            algorithm = "nn"

        if model_id is None:
            model_id = self.active_model_id

        explainer = SimilarExamples(
            self.models[model_id],
            transformers=self.transformers,
            feature_descriptions=self.feature_descriptions,
            e_algorithm=algorithm,
            classes=self.classes,
            class_descriptions=self.class_descriptions,
            training_size=training_size,
            standardize=standardize,
            fast=fast,
        )
        x_train, training_ids = self._get_x_train_orig(x_train_orig, return_ids=True)
        explainer.fit(x_train, self._get_y_train(y_train))
        self.training_ids_for_se = training_ids
        self._add_explainer("se", algorithm, explainer)
        return explainer

    def produce_similar_examples(
        self,
        x_orig,
        model_id=None,
        x_train_orig=None,
        y_train=None,
        format_output=True,
        num_examples=3,
        standardize=False,
        fast=True,
        format_y=True,
        algorithm=None,
        force_refit=False,
    ):
        """
        Produce a SimilarExamples explainer

        Args:
            x_orig (DataFrame):
                Input to explain
            model_id (string or int):
                ID of model to explain
            x_train_orig (DataFrame):
                Data to fit on, if not provided during initialization
            y_train (DataFrame or Series):
                Training targets to fit on, if not provided during initialization
            format_output (Boolean):
                No functionality, included for consistency
            num_examples (int):
                Number of similar examples to return
            standardize (Boolean):
                If True, standardize data before using it to get similar examples.
                Recommended if model-ready data is not already standardized
            fast (Boolean):
                If True, use a faster algorithm for generating similar examples. Disable if
                faiss is not available
            format_y (Boolean):
                If True, format the ground truth y values returned using self.pred_format_func
            algorithm (string):
                Name of algorithm
            force_refit (Boolean):
                If True, initialize and fit a new explainer even if the appropriate explainer
                already exists

        Returns:
            {"X": DataFrame, "y": Series, "Input": Series} (if series),
                else {"id" -> {"X": DataFrame, "y": Series, "Input": Series}}
            X is the examples, ordered from top to bottom by similarity to input and
            y is the corresponding y values
            Input is the original input in the same feature space
        """
        if algorithm is None:
            algorithm = "nn"

        format_kwargs = dict()
        if format_y:
            format_kwargs["y_format_func"] = self.pred_format_func
            format_kwargs["training_ids"] = self.training_ids_for_se
            format_kwargs["id_column_name"] = self.id_column

        return self._produce_explanation_helper(
            "se",
            algorithm,
            self.prepare_similar_examples,
            format_similar_examples_output,
            x_orig=x_orig,
            model_id=model_id,
            x_train_orig=x_train_orig,
            y_train=y_train,
            format_output=format_output,
            force_refit=force_refit,
            prepare_kwargs={"standardize": standardize, "fast": fast},
            produce_kwargs={"num_examples": num_examples},
            format_kwargs=format_kwargs,
        )

    def train_feature_contribution_llm(
        self,
        transformer=None,
        x_train_orig=None,
        live=True,
        provide_examples=False,
        num_inputs=5,
        num_features=3,
    ):
        """
        Run the training process for the LLM model used to generate narrative feature
        contribution explanations.

        Args:
            transformer (NarrativeTransformer):
                NarrativeTransformer to train. If None, this RealApp object will save the
                training data for use in its produce_narrative functions
            x_train_orig (DataFrame of shape (n_instances, n_features)):
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
        lfc_explainers = self._get_explainer("lfc")
        if not lfc_explainers:
            self.prepare_feature_contributions(x_train_orig=x_train_orig, algorithm="shap")
            lfc_explainers = self._get_explainer("lfc")
        training_examples = None
        for i, algorithm in enumerate(lfc_explainers):
            if i == 0:
                training_examples = lfc_explainers[algorithm].train_llm(
                    x_train=self._get_x_train_orig(x_train_orig),
                    live=live,
                    provide_examples=provide_examples,
                    num_inputs=num_inputs,
                    num_features=num_features,
                )
            else:
                lfc_explainers[algorithm].set_llm_training_data(training_data=training_examples)
        if transformer is not None:
            transformer.set_training_examples(
                "feature_contributions", training_examples, replace=True
            )

    def set_openai_client(self, openai_client=None, openai_api_key=None):
        """
        Set the openai client for this RealApp.
        One of openai_client or openai_api_key must be provided.

        Args:
            openai_client (openai.Client):
                OpenAI client object, with API key already set. If provided, openai_api_key is
                ignored
            openai_api_key (string):
                OpenAI API key. If provided, create a new API client.
        """
        if openai_client is not None:
            self.openai_client = openai_client
        elif openai_api_key is not None:
            self.openai_client = OpenAI(api_key=openai_api_key)
        else:
            raise ValueError("Must provide openai_client or openai_api_key")

    @staticmethod
    def from_sklearn(
        pipeline=None,
        model=None,
        transformers=None,
        X_train=None,
        y_train=None,
        refit_model=True,
        verbose=0,
        **kwargs
    ):
        """
        Create a RealApp from a sklearn pipeline or model and transformers.
        Must provide one of:
            - just pipeline
            - just model
            - model and transformers

        Args:
            pipeline (sklearn.pipeline.Pipeline):
                Sklearn pipeline to convert. The final step of the pipeline must be a model.
            model (sklearn model):
                Sklearn model to use. Ignored if pipeline is not None
            transformers (list of sklearn transformers):
                List of sklearn transformers to use. Ignored if pipeline is not None
            X_train (DataFrame):
                Training data to fit transformers and explanations to. May be required if
                transformers are not fitted or must be recreated. If not provided, must be
                provided when preparing and using realapp explainers.
            y_train (DataFrame or Series):
                Training targets to fit transformers and explanations to. If not provided, must be
                provided when preparing and using realapp explainers.
            refit_model (bool):
                If True, refit the model using the new Pyreal transformers. This may be necessary
                as sklearn and Pyreal transformers may result in an unaligned column order.
                Requires X_train and y_train to be provided.
            verbose (int):
                Verbosity level. If 0, no output. If 1, detailed output
            **kwargs:
                Additional arguments to pass to RealApp constructor.

        Returns:
            RealApp
                Newly created RealApp object
        """
        if pipeline is None and model is None and transformers is None:
            raise ValueError("Must provide either pipeline or model")

        if pipeline is not None and not hasattr(pipeline, "steps"):
            raise ValueError("pipeline must be a valid sklearn pipeline")

        pyreal_transformers = []
        if pipeline is not None:
            model = pipeline.steps[-1][1]
            pyreal_transformers = sklearn_pipeline_to_pyreal_transformers(
                pipeline, X_train, verbose=verbose
            )
        elif transformers is not None:
            pyreal_transformers = sklearn_pipeline_to_pyreal_transformers(
                transformers, X_train, verbose=verbose
            )
        if refit_model:
            if X_train is None or y_train is None:
                raise ValueError("X_train and y_train must be provided to refit the model")
            model.fit(run_transformers(pyreal_transformers, X_train), y_train)
        return RealApp(
            models=model,
            transformers=pyreal_transformers,
            X_train_orig=X_train,
            y_train=y_train,
            **kwargs
        )

    def _get_x_train_orig(self, x_train_orig, return_ids=False):
        """
        Helper function to get the appropriate x_orig or raise errors if something goes wrong
        Args:
            x_train_orig (DataFrame or None):
                Provided DataFrame
        Returns:
            The dataframe to use (x_orig or self.x_train_orig), may be None if neither is given
        """
        if x_train_orig is not None:
            if self.id_column is not None and self.id_column in x_train_orig:
                if return_ids:
                    ids = x_train_orig[self.id_column]
                    return x_train_orig.drop(columns=self.id_column), ids
                return x_train_orig.drop(columns=self.id_column)
            if return_ids:
                return x_train_orig, None
            return x_train_orig
        else:
            if return_ids:
                return self.X_train_orig, self.training_ids_for_se
            return self.X_train_orig

    def _get_y_train(self, y_train):
        """
        Helper function to get the appropriate y or raise errors if something goes wrong
        Args:
            y (DataFrame or None):
                Provided DataFrame
        Returns:
            The dataframe to use (y or self.y_train), may be None if neither is given
        """
        if y_train is not None:
            return y_train
        else:
            return self.y_train
