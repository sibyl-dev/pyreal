from abc import ABC, abstractmethod

from pyreal.utils.explanation_algorithm import ExplanationAlgorithm


def fit_transformers(transformers, x):
    """
    Fit a set of transformers in-place, transforming the data after each fit. Checks if each
    transformer has a fit function and if so, calls it.
    Args:
        transformers (list of Transformers):
            List of transformers to fit, in order
        x (DataFrame of shape (n_instances, n_features)):
            Dataset to fit on.

    Returns:
        None
    """
    x_transform = x.copy()
    for transformer in transformers:
        fit_func = getattr(transformer, "fit", None)
        if callable(fit_func):
            fit_func(x_transform)
        x_transform = transformer.transform(x_transform)


def run_transformers(transformers, x):
    """
    Run a series of transformers on x_orig

    Args:
        transformers (list of Transformers):
            List of transformers to fit, in order
        x (DataFrame of shape (n_instances, n_features)):
            Dataset to transform

    Returns:
        DataFrame of shape (n_instances, n_features)
            Transformed data
    """
    x_transform = x.copy()
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

    def transform_explanation(self, explanation, algorithm):
        if algorithm == ExplanationAlgorithm.SHAP:
            return self.transform_explanation_shap(explanation)
        if algorithm == ExplanationAlgorithm.PERMUTATION_IMPORTANCE:
            return self.transform_explanation_permutation_importance(explanation)
        if algorithm == ExplanationAlgorithm.SURROGATE_DECISION_TREE:
            raise NotImplementedError("Explanation transformers do not yet support "
                                      "DecisionTreeExplainer")
        raise ValueError("Invalid algorithm %s" % algorithm)

    # noinspection PyMethodMayBeStatic
    def transform_explanation_shap(self, explanation):
        return explanation

    # noinspection PyMethodMayBeStatic
    def transform_explanation_permutation_importance(self, explanation):
        return explanation
