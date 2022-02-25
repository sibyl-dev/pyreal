from abc import ABC, abstractmethod
import pandas as pd

from pyreal.types.explanations.dataframe import (
    AdditiveFeatureContributionExplanation, AdditiveFeatureImportanceExplanation,
    FeatureContributionExplanation, FeatureImportanceExplanation,)


class BreakingTransformError(Exception):
    """
    Raised in a transform_explanation or inverse_transform_explanation function would be impossible
    and is expected to break further transforms. The explanation transformation process will stop
    upon encountering this error.
    """


def fit_transformers(transformers, x):
    """
    Fit a set of transformers in-place, transforming the data after each fit. Checks if each
    transformer has a fit function and if so, calls it. Returns the data after being transformed
    by the final transformer.
    Args:
        transformers (Transformer or list of Transformers):
            List of transformers to fit, in order
        x (DataFrame of shape (n_instances, n_features)):
            Dataset to fit on.

    Returns:
        DataFrame of shape (n_instances, n_features)
            `x` after being transformed by all transformers
    """
    x_transform = x.copy()
    if not isinstance(transformers, list):
        transformers = [transformers]
    for transformer in transformers:
        fit_func = getattr(transformer, "fit", None)
        if callable(fit_func):
            fit_func(x_transform)
        x_transform = transformer.transform(x_transform)
    return x_transform


def run_transformers(transformers, x):
    """
    Run a series of transformers on x_orig

    Args:
        transformers (Transformer or list of Transformers):
            List of transformers to fit, in order
        x (DataFrame of shape (n_instances, n_features)):
            Dataset to transform

    Returns:
        DataFrame of shape (n_instances, n_features)
            Transformed data
    """
    x_transform = x.copy()
    series = False
    name = None
    if isinstance(x_transform, pd.Series):
        name = x_transform.name
        x_transform = x_transform.to_frame().T
        series = True
    if not isinstance(transformers, list):
        transformers = [transformers]
    for transform in transformers:
        x_transform = transform.transform(x_transform)
    if series:
        x_transform = x_transform.squeeze()
        x_transform.name = name
    return x_transform


def _display_missing_transform_info(transformer_name, function_name):
    print("Transformer %s does not have an implemented %s function. "
          "Defaulting to no change in explanation. If this causes a break,"
          "you may want to add a interpret=False flag to this transformer or redefine this "
          "function to throw a BreakingTransformError."
          % (transformer_name, function_name))


def _display_missing_transform_info_inverse(transformer_name, function_name):
    print("Transformer %s does not have an implemented %s function. "
          "Defaulting to no change in explanation. If this causes a break,"
          "you may want to add an interpret=True flag to this transformer or redefine this "
          "function to throw a BreakingTransformError."
          % (transformer_name, function_name))


class Transformer(ABC):
    """
    An abstract base class for Transformers. Transformers transform data from a first feature space
    to a second, and explanations from the second back to the first.
    """

    def __init__(self, model=True, interpret=False, algorithm=None):
        """
        Set this Transformer's flags.

        Args:
            model (Boolean):
                If True, this transformer is required by the model-ready feature space. It will be
                run any time a model prediction is needed
            interpret (Boolean):
                If True, this transformer makes the data more human-interpretable
            algorithm (Boolean):
                If True, this transformer is required for the explanation algorithm. If
                algorithm is False, but model is True, this transformer will be applied only
                when making model predictions during the explanation algorithm. Cannot be True if
                if the model flag is False
        """
        self.model = model
        self.interpret = interpret
        if algorithm is None:
            self.algorithm = model
        else:
            self.algorithm = algorithm
        if self.model is False and self.algorithm is True:
            raise ValueError("algorithm flag cannot be True if model flag is False")

    def set_flags(self, model=None, interpret=None, algorithm=None):
        if model is not None:
            self.model = model
        if interpret is not None:
            self.interpret = interpret
        if algorithm is not None:
            self.algorithm = algorithm
        if self.model is False and self.algorithm is True:
            raise ValueError("algorithm flag cannot be True if model flag is False")

    def fit(self, x, **params):
        """
        Fit this transformer to data

        Args:
            x (DataFrame of shape (n_instances, n_features)):
                The dataset to fit to
            **params:
                Additional transformer parameters

        Returns:
            None
        """
        return self

    @abstractmethod
    def data_transform(self, x):
        """
        Transforms data `x` to a new feature space.
        Args:
            x (DataFrame of shape (n_instances, n_features)):
                The dataset to transform

        Returns:
            DataFrame of shape (n_instances, n_transformed_features):
                The transformed dataset
        """

    def transform(self, x):
        """
        Wrapper for data_transform.
        Transforms data `x` to a new feature space.

        Included for compatibility with existing ML libraries.
        Args:
            x (DataFrame of shape (n_instances, n_features)):
                The dataset to transform

        Returns:
            DataFrame of shape (n_instances, n_transformed_features):
                The transformed dataset
        """
        return self.data_transform(x)

    def fit_transform(self, x, **fit_params):
        """
        Fits this transformer to data and then transforms the same data

        Args:
            x (DataFrame of shape (n_instances, n_features)):
                The dataset to fit and transform
            **fit_params:
                Parameters for the fit function

        Returns:
            (DataFrame of shape (n_instances, n_transformed_features):
                The transformed dataset
        """
        self.fit(x, **fit_params)
        return self.data_transform(x)

    def inverse_transform_explanation(self, explanation):
        """
        Transforms the explanation from the second feature space handled by this transformer
        to the first.

        Args:
            explanation (Explanation):
                The explanation to transform
        Returns:
            Explanation:
                The transformed explanation
        Raises:
            ValueError
                If `explanation` is not of a supported ExplanationType

        """
        if isinstance(explanation, AdditiveFeatureContributionExplanation) \
                or isinstance(explanation, AdditiveFeatureImportanceExplanation):
            return self.inverse_transform_explanation_additive_contributions(explanation)
        # TODO: here we are temporarily using the additive version for non-additive explanations
        #       Addressed in GH issue 114.
        if isinstance(explanation, FeatureContributionExplanation):
            return self.inverse_transform_explanation_additive_contributions(explanation)
        if isinstance(explanation, FeatureImportanceExplanation):
            return self.inverse_transform_explanation_feature_importance(explanation)
        raise ValueError("Invalid explanation types %s" % explanation.__class__)

    def transform_explanation(self, explanation):
        """
        Transforms the explanation from the first feature space handled by this transformer
        to the second.

        Args:
            explanation (Explanation):
                The explanation to transform
        Returns:
            Explanation:
                The transformed explanation
        Raises:
            ValueError
                If `explanation` is not of a supported ExplanationType

        """
        if isinstance(explanation, AdditiveFeatureContributionExplanation) \
                or isinstance(explanation, AdditiveFeatureImportanceExplanation):
            return self.transform_explanation_additive_contributions(explanation)
        # for now, use the additive version for non-additive explanations
        if isinstance(explanation, FeatureContributionExplanation):
            return self.transform_explanation_additive_contributions(explanation)
        if isinstance(explanation, FeatureImportanceExplanation):
            return self.transform_explanation_feature_importance(explanation)
        raise ValueError("Invalid explanation types %s" % explanation.__class__)

    # noinspection PyMethodMayBeStatic
    def inverse_transform_explanation_additive_contributions(self, explanation):
        """
        Transforms additive contribution explanations

        Args:
            explanation (AdditiveFeatureContributionExplanationType):
                The explanation to be transformed

        Returns:
            AdditiveFeatureContributionExplanationType:
                The transformed explanation
        """
        _display_missing_transform_info_inverse(
            self.__class__, "inverse_transform_explanation_additive_contributions")
        return explanation

    # noinspection PyMethodMayBeStatic
    def inverse_transform_explanation_feature_importance(self, explanation):
        """
        Transforms feature importance explanations

        Args:
            explanation (FeatureImportanceExplanationType):
                The explanation to be transformed
        Returns:
            FeatureImportanceExplanationType:
                The transformed explanation
        """
        _display_missing_transform_info_inverse(
            self.__class__, "inverse_transform_explanation_feature_importance")
        return explanation

    # noinspection PyMethodMayBeStatic
    def transform_explanation_additive_contributions(self, explanation):
        """
        Transforms additive contribution explanations

        Args:
            explanation (AdditiveFeatureContributionExplanationType):
                The explanation to be transformed

        Returns:
            AdditiveFeatureContributionExplanationType:
                The transformed explanation
        """
        _display_missing_transform_info(
            self.__class__, "transform_explanation_additive_contributions")
        return explanation

    # noinspection PyMethodMayBeStatic
    def transform_explanation_feature_importance(self, explanation):
        """
        Transforms feature importance explanations

        Args:
            explanation (FeatureImportanceExplanationType):
                The explanation to be transformed
        Returns:
            FeatureImportanceExplanationType:
                The transformed explanation
        """
        _display_missing_transform_info(
            self.__class__, "transform_explanation_feature_importance")
        return explanation
