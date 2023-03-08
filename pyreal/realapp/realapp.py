from pyreal.explainers import Explainer


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
            model_id: self._make_explainer(self.models[model_id]) for model_id in self.models
        }

        self.explainers = {}

    def _make_explainer(self, model):
        return Explainer(
            model,
            self.X_train_orig,
            y_orig=self.y_orig,
            transformers=self.transformers,
            feature_descriptions=self.feature_descriptions,
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

    def predict(self, x, model_id=None):
        if model_id is None:
            model_id = self.active_model_id

        return self.base_explainers[model_id].model_predict(x)
