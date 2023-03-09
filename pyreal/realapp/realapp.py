import pandas as pd
import numpy as np

from pyreal.explainers import Explainer, LocalFeatureContribution


def _parse_feature_contribution_df(contributions, values, average_mode):
    """
    Convert the contributions and values into the expected output format,

    Args:
        contributions (Series of shape (n_features,)):
            The contributions, with feature names as indices
        values (Series of shape (n_features, )):
            The values, with feature names as indices

    Returns:
        One dataframe, with each row representing a feature, and four columns:
            Feature Name    Feature Value   Contribution    Average/Mode
    """
    feature_names = contributions.index
    df = pd.DataFrame(
        {
            "Feature Name": feature_names,
            "Feature Value": values,
            "Contribution": contributions,
            "Average/Mode": average_mode,
        }
    )
    return df


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
    return df.drop(s.index, axis=1).mode().iloc[0].append(s)


class RealApp:
    """
    Maintains all information about a Pyreal application to generate explanations
    """

    def __init__(
        self,
        models,
        X_train_orig,
        y_orig=None,
        transformers=None,
        feature_descriptions=None,
        active_model_id=None,
        classes=None,
        class_descriptions=None,
    ):
        """
        Initialize a RealApp object

        Args:
            models (model object, list of models, or dict of model_id:model):
                Model(s) for this application
            X_train_orig (DataFrame of shape (n_instances,n_features):
                Training data for models
            y_orig (DataFrame of shape (n_instances,)):
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

        self.X_train_orig = X_train_orig
        self.y_orig = y_orig

        if isinstance(transformers, list):
            self.transformers = transformers
        else:  # assume single transformer given
            self.transformers = [transformers]
        self.transformers = transformers
        self.feature_descriptions = feature_descriptions

        # Base explainer used for general transformations and model predictions
        # Also validates data, model, and transformers
        self.base_explainers = {
            model_id: self._make_base_explainer(self.models[model_id]) for model_id in self.models
        }

        self.explainers = {}  # Dictionary of dictionaries:
        # {"explanation_type": {"algorithm":Explainer} }

    def _make_base_explainer(self, model):
        return Explainer(
            model,
            self.X_train_orig,
            y_orig=self.y_orig,
            transformers=self.transformers,
            feature_descriptions=self.feature_descriptions,
        )

    def _explainer_exists(self, explanation_type, algorithm):
        if explanation_type in self.explainers:
            if algorithm in self.explainers[explanation_type]:
                return True
        return False

    def _add_explainer(self, explanation_type, algorithm, explainer):
        if explanation_type not in self.explainers:
            self.explainers[explanation_type] = {}
        self.explainers[explanation_type][algorithm] = explainer

    def _get_explainer(self, explanation_type, algorithm):
        return self.explainers[explanation_type, algorithm]

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

    def predict(self, x, model_id=None):
        """
        Predict on x using the active model or model specified by model_id

        Args:
            x (DataFrame of shape (n_instances, n_features)):
                Data to predict on
            model_id (int or string):
                Model to use for prediction

        Returns:
            (model return type)
                Model prediction on x
        """
        if model_id is None:
            model_id = self.active_model_id

        return self.base_explainers[model_id].model_predict(x)

    def produce_local_feature_contributions(
        self, x_orig, model_id=None, algorithm=None, id_column_name=None, shap_type=None
    ):
        if model_id is None:
            model_id = self.active_model_id

        if algorithm is None:
            algorithm = "shap"

        if self._explainer_exists("lfc", algorithm):
            explainer = self._get_explainer("lfc", algorithm)
        else:
            explainer = LocalFeatureContribution(
                self.models[model_id],
                self.X_train_orig,
                e_algorithm=algorithm,
                shap_type=shap_type,
                fit_on_init=True,
            )
            self._add_explainer("lfc", algorithm, explainer)

        if id_column_name is not None:
            ids = x_orig[id_column_name]
            x_orig = x_orig.drop(columns=id_column_name)
        else:
            ids = x_orig.index

        explanation = explainer.produce(x_orig)
        average_mode = _get_average_or_mode(explanation.get())
        explanation_dict = {}
        for i, row_id in enumerate(ids):
            explanation_dict[row_id] = _parse_feature_contribution_df(
                explanation.get().iloc[i, :], explanation.get_values().iloc[i, :], average_mode
            )

        return explanation_dict
