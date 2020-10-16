from abc import ABC
from abc import abstractmethod

import pandas as pd

from real.utils import model_utils
from real.utils.transformer import run_transformers


def _check_transforms(transforms):
    if transforms is None:
        return None
    if not isinstance(transforms, list):
        transforms = [transforms]
    else:
        transforms = transforms
    for transformer in transforms:
        transform_method = getattr(transformer, "transform", None)
        if not callable(transform_method):
            raise ValueError("Given transformer that does not have a .transform function")
    return transforms


class Explainer(ABC):
    """
    Generic Explainer object

    Args:
        model (string filepath or model object):
           Filepath to the pickled model to explain, or model object with .predict() function
        x_orig (dataframe of shape (n_instances, x_orig_feature_count)):
           The training set for the explainer
        y_orig (dataframe of shape (n_instances,-)):
           The y values for the dataset
        feature_descriptions (dict):
           Interpretable descriptions of each feature
        e_transforms (transformer object or list of transformer objects):
           Transformer(s) that need to be used on x_orig for the explanation algorithm:
           x_orig -> x_explain
        m_transforms (transformer object or list of transformer objects):
           Transformer(s) needed on x_orig to make predictions on the dataset with model,
           if different than e_transforms
           x_orig -> x_model
        i_transforms (transformer object or list of transformer objects):
           Transformer(s) needed to make x_orig interpretable
           x_orig -> x_interpret
        fit_on_init (Boolean):
           If True, fit the explainer on initiation.
           If False, self.fit() must be manually called before produce() is called
    """
    def __init__(self, model,
                 x_orig, y_orig=None,
                 feature_descriptions=None,
                 e_transforms=None, m_transforms=None, i_transforms=None,
                 fit_on_init=False):
        if isinstance(model, str):
            self.model = model_utils.load_model_from_pickle(model)
        else:
            # TODO: confirm model is valid
            self.model = model

        self.X_orig = x_orig
        self.y_orig = y_orig

        if not isinstance(x_orig, pd.DataFrame) or \
                (y_orig is not None and not isinstance(y_orig, pd.DataFrame)):
            raise TypeError("X_orig and y_orig must be of type DataFrame")

        self.expected_feature_number = x_orig.shape[1]

        self.x_orig_feature_count = x_orig.shape[1]

        self.e_transforms = _check_transforms(e_transforms)
        self.m_transforms = _check_transforms(m_transforms)
        self.i_transforms = _check_transforms(i_transforms)

        self.feature_descriptions = feature_descriptions

        if fit_on_init:
            self.fit()

    @abstractmethod
    def fit(self):
        """
        Fit this explainer object. Abstract method
        """
        pass

    @abstractmethod
    def produce(self, x_orig):
        """
        Return the explanation. Abstract method

        Args:
            x_orig (DataFrame of shape (n_instances, n_features):
                Input to explain

        Returns:
            Type varies by subclass
                Explanation
        """
        pass

    def transform_to_x_explain(self, x_orig):
        """
        Transform x_orig to x_explain, using the e_transforms

        Args:
            x_orig (DataFrame of shape (n_instances, x_orig_feature_count)):
                Original input
        Returns:
             DataFrame of shape (n_instances, x_explain_feature_count)
                x_orig converted to explainable form
        """
        if self.e_transforms is None:
            return x_orig
        return run_transformers(self.e_transforms, x_orig)

    def transform_to_x_model(self, x_orig):
        """
        Transform x_orig to x_model, using the m_transforms

        Args:
            x_orig (DataFrame of shape (n_instances, x_orig_feature_count)):
                Original input

        Returns:
             DataFrame of shape (n_instances, x_model_feature_count)
                x_orig converted to model-ready form
        """
        if self.m_transforms is None:
            return x_orig
        return run_transformers(self.m_transforms, x_orig)

    def transform_to_x_interpret(self, x_orig):
        """
        Transform x_orig to x_interpret, using the i_transforms
        Args:
            x_orig (DataFrame of shape (n_instances, x_orig_feature_count)):
                Original input

        Returns:
             DataFrame of shape (n_instances, x_interpret_feature_count)
                x_orig converted to interpretable form
        """
        if self.i_transforms is None:
            return x_orig
        return run_transformers(self.i_transforms, x_orig)

    def model_predict(self, x_orig):
        """
        Predict on x_orig using the model and return the result

        Args:
            x_orig (DataFrame of shape (n_instances, x_orig_feature_count)):
                Data to predict on

        Returns:
            model output type
                Model prediction
        """
        if x_orig.ndim == 1:
            x_orig = x_orig.to_frame().T
        x_model = self.transform_to_x_model(x_orig)
        return self.model.predict(x_model)

    def feature_description(self, feature_name):
        """
        Returns the interpretable description associated with a feature

        Args:
            feature_name (string)

        Returns:
            string
                 Description of feature
        """
        return self.feature_descriptions[feature_name]

    def convert_columns_to_interpretable(self, df):
        if self.feature_descriptions is None:
            # TODO: log a warning
            return df
        return df.rename(self.feature_descriptions, axis="columns")

    def convert_data_to_interpretable(self, x_orig):
        """
        Convert data in its original form to an interpretable form, with interpretable features
        Args:
            x_orig (DataFrame of shape (n_instances, n_features)):
                Input data to convert

        Returns:
            DataFrame of shape (n_instances, x_interpret_feature_count)
                Transformed, interpretable data
        """
        return self.convert_columns_to_interpretable(self.transform_to_x_interpret(x_orig))