import numpy as np
import pandas as pd

from pyreal.explainers import (
    Explainer,
    GlobalFeatureImportance,
    LocalFeatureContribution,
    SimilarExamples,
)
from pyreal.utils import get_top_contributors


def format_feature_contribution_output(explanation, ids=None, series=False, optimized=False):
    """
    Format Pyreal FeatureContributionExplanation objects into Local Feature Contribution outputs
    Args:
        explanation (FeatureContributionExplanation):
            Pyreal Explanation object to parse
        ids (list of strings or ints):
            List of row ids
        series (Boolean):
            If True, the produce function was passed a series input
        optimized (Boolean)
            If True, return in a simple DataFrame format

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
    average_mode = _get_average_or_mode(explanation.get_values())
    explanation_dict = {}
    for i, row_id in enumerate(ids):
        contributions = explanation.get().iloc[i, :]
        values = explanation.get_values().iloc[i, :].loc[contributions.index]
        average_mode = average_mode.loc[contributions.index]

        feature_names = contributions.index

        explanation_dict[row_id] = pd.DataFrame.from_dict(
            {
                "Feature Name": feature_names.values,
                "Feature Value": values.values,
                "Contribution": contributions.values,
                "Average/Mode": average_mode.values,
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
    return pd.DataFrame({"Feature Name": importances.columns, "Importance": importances.squeeze()})


def format_similar_examples_output(
    explanation, ids=None, series=False, y_format_func=None, optimized=False
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
    for key, row_id in enumerate(ids):
        examples = explanation.get_examples(row_id=key)
        targets = explanation.get_targets(row_id=key)
        if y_format_func is not None:
            targets = targets.apply(y_format_func)
        result[row_id] = {
            "X": examples,
            "y": targets,
            "Input": explanation.get_values().iloc[key, :],
        }
    if series:
        return result[next(iter(result))]
    return result


def _get_average_or_mode(df):
    """
    Gets the average of numeric features and the mode of categorical features

    Args:
        df (DataFrame):
            Input
    Returns:
        Series
            Average or mode of every column in df
    """
    s = df.select_dtypes(np.number).mean()
    if len(s) == df.shape[1]:  # all columns are numeric
        return s
    return pd.concat((df.drop(s.index, axis=1).mode().iloc[0], s))


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

        if (
            X_train_orig is not None
            and self.id_column is not None
            and self.id_column in X_train_orig
        ):
            self.X_train_orig = X_train_orig.drop(columns=self.id_column)
        else:
            self.X_train_orig = X_train_orig
        self.y_train = y_train

        self.classes = classes
        self.class_descriptions = class_descriptions
        self.pred_format_func = pred_format_func

        if isinstance(transformers, list):
            self.transformers = transformers
        else:  # assume single transformer given
            self.transformers = [transformers]
        self.transformers = transformers
        self.feature_descriptions = feature_descriptions

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

    def _get_explainer(self, explanation_type, algorithm):
        """
        Get the requested explainer

        Args:
            explanation_type (string):
                Code for explanation_type
            algorithm (string):
                Name of algorithm

        Returns:
            Explainer
                The requested explainer
        """
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

        if x_orig is not None:
            series = x_orig.ndim == 1
            ids = None

            if self.id_column is not None and self.id_column in x_orig:
                ids = x_orig[self.id_column]
                if series:  # If x was a series, ids will now be a scaler
                    ids = [ids]
                x_orig = x_orig.drop(self.id_column, axis=x_orig.ndim - 1)

            explanation = explainer.produce(x_orig, **produce_kwargs)
            return format_output_func(
                explanation, ids, optimized=not format_output, series=series, **format_kwargs
            )
        else:
            explanation = explainer.produce(**produce_kwargs)
            return format_output_func(explanation, optimized=not format_output, **format_kwargs)

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
        explainer.fit(self._get_x_train_orig(x_train_orig), self._get_y_train(y_train))
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
            DataFrame with a Feature Name column and an Importance column
        """
        if algorithm is None:
            algorithm = "nn"

        format_kwargs = dict()
        if format_y:
            format_kwargs["y_format_func"] = self.pred_format_func

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

    def _get_x_train_orig(self, x_train_orig):
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
                return x_train_orig.drop(columns=self.id_column)
            return x_train_orig
        else:
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
