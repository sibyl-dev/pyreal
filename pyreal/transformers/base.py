from abc import ABC, abstractmethod

from pyreal.types.explanations.dataframe import (
    AdditiveFeatureContributionExplanationType, AdditiveFeatureImportanceExplanationType,
    FeatureImportanceExplanationType,)


def fit_transformers(transformers, x_orig):
    """
    Fit a set of transformers in-place, transforming the data after each fit. Checks if each
    transformer has a fit function and if so, calls it.
    Args:
        transformers (list of Transformers):
            List of transformers to fit, in order
        x_orig (DataFrame of shape (n_instances, n_features)):
            Dataset to fit on.

    Returns:
        None
    """
    x_transform = x_orig.copy()
    for transformer in transformers:
        fit_func = getattr(transformer, "fit", None)
        if callable(fit_func):
            fit_func(x_transform)
        x_transform = transformer.transform(x_transform)


def run_transformers(transformers, x_orig):
    """
    Run a series of transformers on x_orig

    Args:
        transformers (list of Transformers):
            List of transformers to fit, in order
        x_orig (DataFrame of shape (n_instances, n_features)):
            Dataset to transform

    Returns:
        DataFrame of shape (n_instances, n_features)
            Transformed data
    """
    x_transform = x_orig.copy()
    for transform in transformers:
        x_transform = transform.transform(x_transform)
    return x_transform


class BaseTransformer(ABC):
    @abstractmethod
    def transform(self, x):
        pass

    def fit(self, x, **params):
        pass

    def fit_transform(self, x, **fit_params):
        self.fit(x, **fit_params)
        return self.transform(x)

    def transform_explanation(self, explanation):
        if isinstance(explanation, AdditiveFeatureContributionExplanationType) \
                or isinstance(explanation, AdditiveFeatureImportanceExplanationType):
            return self.transform_explanation_additive_contributions(explanation)
        if isinstance(explanation, FeatureImportanceExplanationType):
            return self.transform_explanation_feature_importance(explanation)
        raise ValueError("Invalid explanation types %s" % explanation.__class__)

    # noinspection PyMethodMayBeStatic
    def transform_explanation_additive_contributions(self, explanation):
        return explanation

    # noinspection PyMethodMayBeStatic
    def transform_explanation_feature_importance(self, explanation):
        return explanation
