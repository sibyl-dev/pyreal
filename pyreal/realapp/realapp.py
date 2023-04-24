import numpy as np
import pandas as pd

from pyreal.explainers import Explainer, GlobalFeatureImportance, LocalFeatureContribution


def format_feature_contribution_output(explanation, ids=None):
    """
    Format Pyreal FeatureContributionExplanation objects into Local Feature Contribution outputs
    Args:
        explanation (FeatureContributionExplanation):
            Pyreal Explanation object to parse
        ids (list of strings or ints):
            List of row ids

    Returns:
        One dataframe per id, with each row representing a feature, and four columns:
            Feature Name    Feature Value   Contribution    Average/Mode
    """
    if ids is None:
        ids = explanation.get().index
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
    return explanation_dict


def format_feature_importance_output(explanation):
    """
    Format Pyreal FeatureImportanceExplanation objects into Global Feature Importance outputs
    Args:
        explanation (FeatureImportanceExplanation):
            Pyreal Explanation object to parse

    Returns:
        DataFrame with a Feature Name column and an Importance column
    """
    importances = explanation.get()
    return pd.DataFrame({"Feature Name": importances.columns, "Importance": importances.squeeze()})


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
    # return df.drop(s.index, axis=1).mode().iloc[0].append(s)


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
                X_train_orig,
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
        x_train_orig=None,
        y_train=None,
        x_orig=None,
        model_id=None,
        force_refit=False,
        **kwargs
    ):
        """
        Produce an explanation from a specified Explainer

        Args:
            explanation_type (string):
                Code for explanation_type
            algorithm (string):
                Name of algorithm
            prepare_explainer_func (function):
                Function that initializes and fits the appropriate explainer
            format_output_func (function):
                Function that formats Explanation objects into the appropriate output format
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
            **kwargs:
                Additional explainer parameters

        Returns:
            Type varies by explanation type
                The explanation
        """
        if model_id is None:
            model_id = self.active_model_id

        if self._explainer_exists(explanation_type_code, algorithm) and not force_refit:
            explainer = self._get_explainer(explanation_type_code, algorithm)
        else:
            explainer = prepare_explainer_func(
                model_id=model_id,
                algorithm=algorithm,
                x_train_orig=x_train_orig,
                y_train=y_train,
                **kwargs
            )

        if x_orig is not None:
            ids = None

            if self.id_column is not None and self.id_column in x_orig:
                ids = x_orig[self.id_column]
                x_orig = x_orig.drop(columns=self.id_column)

            explanation = explainer.produce(x_orig)
            return format_output_func(explanation, ids)
        else:
            explanation = explainer.produce()
            return format_output_func(explanation)

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

    def predict(self, x, model_id=None, as_dict=True):
        """
        Predict on x using the active model or model specified by model_id

        Args:
            x (DataFrame of shape (n_instances, n_features)):
                Data to predict on
            model_id (int or string):
                Model to use for prediction
            as_dict (Boolean):
                If False, return predictions as a single Series/List. Otherwise, return
                in {row_id: pred} format.

        Returns:
            (model return type)
                Model prediction on x
        """
        if self.id_column is not None and self.id_column in x:
            ids = x[self.id_column]
            x = x.drop(columns=self.id_column)
        else:
            ids = x.index
        if model_id is None:
            model_id = self.active_model_id

        preds = self.base_explainers[model_id].model_predict(x)
        if not as_dict:
            return preds
        preds_dict = {}
        for i, row_id in enumerate(ids):
            preds_dict[row_id] = preds[i]
        return preds_dict

    def prepare_feature_contributions(
        self,
        model_id=None,
        x_train_orig=None,
        y_train=None,
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
        shap_type=None,
        force_refit=False,
        training_size=None,
    ):
        """
        Produce a feature contribution explanation

        Args:
            x_orig (DataFrame):
                Input to explain
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

        Returns:
            One dataframe per id, with each row representing a feature, and four columns:
            Feature Name    Feature Value   Contribution    Average/Mode
        """
        if algorithm is None:
            algorithm = "shap"

        return self._produce_explanation_helper(
            "lfc",
            algorithm,
            self.prepare_feature_contributions,
            format_feature_contribution_output,
            x_train_orig=x_train_orig,
            y_train=y_train,
            x_orig=x_orig,
            model_id=model_id,
            force_refit=force_refit,
            shap_type=shap_type,
            training_size=training_size,
        )

    def prepare_feature_importance(
        self,
        model_id=None,
        x_train_orig=None,
        y_train=None,
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
        shap_type=None,
        force_refit=False,
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
            shap_type (string):
                If algorithm="shap", type of SHAP explainer to use
            force_refit (Boolean):
                If True, initialize and fit a new explainer even if the appropriate explainer
                already exists

        Returns:
            DataFrame with a Feature Name column and an Importance column
        """
        if algorithm is None:
            algorithm = "shap"

        return self._produce_explanation_helper(
            "gfi",
            algorithm,
            self.prepare_feature_importance,
            format_feature_importance_output,
            model_id=model_id,
            x_train_orig=x_train_orig,
            y_train=y_train,
            force_refit=force_refit,
            shap_type=shap_type,
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
