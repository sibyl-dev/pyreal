import logging
from abc import ABC, abstractmethod

import pandas as pd

from pyreal.types.explanations.base import Explanation
from pyreal.types.explanations.decision_tree import DecisionTreeExplanation
from pyreal.types.explanations.feature_based import (
    AdditiveFeatureContributionExplanation,
    AdditiveFeatureImportanceExplanation,
    FeatureBased,
    FeatureContributionExplanation,
    FeatureImportanceExplanation,
)

log = logging.getLogger(__name__)


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
    if series and isinstance(x_transform, pd.DataFrame):
        x_transform = x_transform.squeeze()
        x_transform.name = name
    return x_transform


def _display_missing_transform_info(transformer_name, function_name):
    log.info(
        "Transformer %s does not have an implemented %s function. "
        "Defaulting to no change in explanation. If this causes a break,"
        "you may want to add a interpret=False flag to this transformer or redefine this "
        "function to throw a BreakingTransformError." % (transformer_name, function_name)
    )


def _display_missing_transform_info_inverse(transformer_name, function_name):
    log.info(
        "Transformer %s does not have an implemented %s function. "
        "Defaulting to no change in explanation. If this causes a break,"
        "you may want to add an interpret=True flag to this transformer or redefine this "
        "function to throw a BreakingTransformError." % (transformer_name, function_name)
    )


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
        self.fitted = False

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
        self.fitted = True
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
        if isinstance(explanation, AdditiveFeatureContributionExplanation):
            return self.inverse_transform_explanation_additive_feature_contribution(explanation)
        if isinstance(explanation, AdditiveFeatureImportanceExplanation):
            return self.inverse_transform_explanation_additive_feature_importance(explanation)
        if isinstance(explanation, FeatureContributionExplanation):
            return self.inverse_transform_explanation_feature_contribution(explanation)
        if isinstance(explanation, FeatureImportanceExplanation):
            return self.inverse_transform_explanation_feature_importance(explanation)
        if isinstance(explanation, FeatureBased):
            return self.inverse_transform_explanation_feature_based(explanation)

        if isinstance(explanation, DecisionTreeExplanation):
            return self.inverse_transform_explanation_decision_tree(explanation)

        if isinstance(explanation, Explanation):  # handle generic explanation cases
            return explanation

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
        print(explanation)
        if isinstance(explanation, AdditiveFeatureContributionExplanation):
            return self.transform_explanation_additive_feature_contribution(explanation)
        if isinstance(explanation, AdditiveFeatureImportanceExplanation):
            return self.transform_explanation_additive_feature_importance(explanation)
        if isinstance(explanation, FeatureContributionExplanation):
            return self.transform_explanation_feature_contribution(explanation)
        if isinstance(explanation, FeatureImportanceExplanation):
            return self.transform_explanation_feature_importance(explanation)
        if isinstance(explanation, FeatureBased):
            return self.transform_explanation_feature_based(explanation)

        if isinstance(explanation, Explanation):
            return explanation

        raise ValueError("Invalid explanation types %s" % explanation.__class__)

    # ========================== INVERSE TRANSFORM EXPLANATION METHODS ===========================

    # noinspection PyMethodMayBeStatic
    def inverse_transform_explanation_additive_feature_contribution(self, explanation):
        """
        Inverse transforms additive feature contribution explanations

        Args:
            explanation (AdditiveFeatureContributionExplanation):
                The explanation to be transformed

        Returns:
            AdditiveFeatureContributionExplanation:
                The transformed explanation
        """
        return AdditiveFeatureContributionExplanation(
            self.inverse_transform_explanation_feature_contribution(explanation).get()
        )

    # noinspection PyMethodMayBeStatic
    def inverse_transform_explanation_additive_feature_importance(self, explanation):
        """
        Inverse transforms additive feature importance explanations

        Args:
            explanation (AdditiveFeatureImportanceExplanation):
                The explanation to be transformed

        Returns:
            AdditiveFeatureImportanceExplanation:
                The transformed explanation
        """
        return AdditiveFeatureImportanceExplanation(
            self.inverse_transform_explanation_feature_importance(explanation).get()
        )

    # noinspection PyMethodMayBeStatic
    def inverse_transform_explanation_feature_contribution(self, explanation):
        """
        Inverse transforms feature contribution explanations

        Args:
            explanation (FeatureContributionExplanation):
                The explanation to be transformed
        Returns:
            FeatureContributionExplanation:
                The transformed explanation
        """
        return FeatureContributionExplanation(
            self.inverse_transform_explanation_feature_based(explanation).get()
        )

    # noinspection PyMethodMayBeStatic
    def inverse_transform_explanation_feature_importance(self, explanation):
        """
        Inverse transforms feature importance explanations

        Args:
            explanation (FeatureImportanceExplanation):
                The explanation to be transformed
        Returns:
            FeatureImportanceExplanation:
                The transformed explanation
        """
        return FeatureImportanceExplanation(
            self.inverse_transform_explanation_feature_based(explanation).get()
        )

    # noinspection PyMethodMayBeStatic
    def inverse_transform_explanation_feature_based(self, explanation):
        """
        Inverse transforms feature-based explanations

        Args:
            explanation (FeatureBased):
                The explanation to be transformed
        Returns:
            FeatureBased:
                The transformed explanation
        """
        _display_missing_transform_info_inverse(
            self.__class__, "inverse_transform_explanation_feature_based"
        )
        return explanation

    # noinspection PyMethodMayBeStatic
    def inverse_transform_explanation_decision_tree(self, explanation):
        """
        Inverse transforms feature-based explanations

        Args:
            explanation (DecisionTree):
                The explanation to be transformed
        Returns:
            DecisionTree:
                The transformed explanation
        """
        _display_missing_transform_info_inverse(
            self.__class__, "inverse_transform_explanation_decision_tree"
        )
        return explanation

    # ============================== TRANSFORM EXPLANATION METHODS ================================

    # noinspection PyMethodMayBeStatic
    def transform_explanation_additive_feature_contribution(self, explanation):
        """
        Transforms additive feature contribution explanations

        Args:
            explanation (AdditiveFeatureContributionExplanation):
                The explanation to be transformed

        Returns:
            AdditiveFeatureContributionExplanation:
                The transformed explanation
        """
        return AdditiveFeatureContributionExplanation(
            self.transform_explanation_feature_contribution(explanation).get()
        )

    # noinspection PyMethodMayBeStatic
    def transform_explanation_additive_feature_importance(self, explanation):
        """
        Transforms additive feature importance explanations

        Args:
            explanation (AdditiveFeatureImportanceExplanation):
                The explanation to be transformed

        Returns:
            AdditiveFeatureImportanceExplanation:
                The transformed explanation
        """
        return AdditiveFeatureImportanceExplanation(
            self.transform_explanation_feature_importance(explanation).get()
        )

    # noinspection PyMethodMayBeStatic
    def transform_explanation_feature_contribution(self, explanation):
        """
        Transforms feature contribution explanations

        Args:
            explanation (FeatureContributionExplanation):
                The explanation to be transformed
        Returns:
            FeatureContributionExplanation:
                The transformed explanation
        """
        return FeatureContributionExplanation(
            self.transform_explanation_feature_based(explanation).get()
        )

    # noinspection PyMethodMayBeStatic
    def transform_explanation_feature_importance(self, explanation):
        """
        Transforms feature importance explanations

        Args:
            explanation (FeatureImportanceExplanation):
                The explanation to be transformed
        Returns:
            FeatureImportanceExplanation:
                The transformed explanation
        """
        return FeatureImportanceExplanation(
            self.transform_explanation_feature_based(explanation).get()
        )

    # noinspection PyMethodMayBeStatic
    def transform_explanation_feature_based(self, explanation):
        """
        Transforms feature-based explanations

        Args:
            explanation (FeatureBased):
                The explanation to be transformed
        Returns:
            FeatureBased:
                The transformed explanation
        """
        _display_missing_transform_info(self.__class__, "transform_explanation_feature_based")
        return explanation

    # noinspection PyMethodMayBeStatic
    def transform_explanation_decision_tree(self, explanation):
        """
        Inverse transforms feature-based explanations

        Args:
            explanation (DecisionTree):
                The explanation to be transformed
        Returns:
            DecisionTree:
                The transformed explanation
        """
        _display_missing_transform_info(self.__class__, "transform_explanation_decision_tree")
        return explanation
