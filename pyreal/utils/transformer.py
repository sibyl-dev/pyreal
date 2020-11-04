from abc import ABC, abstractmethod
from enum import Enum, auto

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


class ExplanationAlgorithm(Enum):
    SHAP = auto()


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
    def transform(self, data):
        pass

    def fit(self, *args, **kwargs):
        pass

    def transform_explanation(self, explanation, algorithm):
        if algorithm == ExplanationAlgorithm.SHAP:
            return self.transform_explanation_shap(explanation)
        raise ValueError("Invalid algorithm %s" % algorithm)

    # noinspection PyMethodMayBeStatic
    def transform_explanation_shap(self, explanation):
        return explanation


class MappingsEncoderTransformer(BaseTransformer):
    """
    Converts data from categorical form to one-hot-encoded
    """
    def __init__(self, mappings):
        self.mappings = mappings

    def transform(self, data):
        cols = data.columns
        num_rows = data.shape[0]
        ohe_data = {}
        for col in cols:
            values = data[col]
            for item in self.mappings.categorical_to_one_hot[col]:
                new_col_name = item[0]
                ohe_data[new_col_name] = np.zeros(num_rows)
                ohe_data[new_col_name][np.where(values == item[1])] = 1
        return pd.DataFrame(ohe_data)


class MappingsDecoderTransformer(BaseTransformer):
    """
    Converts data from one-hot encoded form to categorical
    """
    def __init__(self, mappings):
        self.mappings = mappings

    def transform(self, data):
        cat_data = {}
        cols = data.columns
        num_rows = data.shape[0]

        for col in cols:
            if col not in self.mappings.one_hot_to_categorical:
                cat_data[col] = data[col]
            else:
                new_name = self.mappings.one_hot_to_categorical[col][0]
                if new_name not in cat_data:
                    cat_data[new_name] = np.empty(num_rows, dtype="object")
                # TODO: add functionality to handle defaults
                cat_data[new_name][np.where(data[col] == 1)] = \
                    self.mappings.one_hot_to_categorical[col][1]
        return pd.DataFrame(cat_data)


class FeatureSelectTransformer(BaseTransformer):
    def __init__(self, feature_names):
        self.feature_names = feature_names

    def transform(self, data):
        return data[self.feature_names]


class OneHotEncoderWrapper(BaseTransformer):
    def __init__(self, feature_list=None):
        self.ohe = OneHotEncoder(sparse=False)
        self.feature_list = feature_list
        self.is_fit = False

    def fit(self, x_orig):
        if self.feature_list is None:
            self.feature_list = x_orig.columns
        self.ohe.fit(x_orig[self.feature_list])
        self.is_fit = True

    def transform(self, x_orig):
        if not self.is_fit:
            raise RuntimeError("Must fit one hot encoder before transforming")
        x_to_encode = x_orig[self.feature_list]
        columns = self.ohe.get_feature_names(x_to_encode.columns)
        index = x_to_encode.index
        x_cat_ohe = self.ohe.transform(x_to_encode)
        x_cat_ohe = pd.DataFrame(x_cat_ohe, columns=columns, index=index)
        return pd.concat([x_orig.drop(self.feature_list, axis="columns"), x_cat_ohe], axis=1)

    def transform_explanation_shap(self, explanation):
        if explanation.ndim == 1:
            explanation = explanation.reshape(1, -1)
        encoded_columns = self.ohe.get_feature_names(self.feature_list)
        for original_feature in self.feature_list:
            encoded_features = [item for item in encoded_columns if
                                item.startswith(original_feature + "_")]
            summed_contribution = explanation[encoded_features].sum(axis=1)
            explanation = explanation.drop(encoded_features, axis="columns")
            explanation[original_feature] = summed_contribution
        return explanation


class DataFrameWrapper(BaseTransformer):
    """
    Allows use of standard sklearn transformers while maintaining DataFrame type.
    """
    def __init__(self, base_transformer):
        self.base_transformer = base_transformer

    def fit(self, x):
        self.base_transformer.fit(x)

    def transform(self, x):
        transformed_np = self.base_transformer.transform(x)
        return pd.DataFrame(transformed_np, columns=x.columns, index=x.index)


class ColumnDropTransformer(BaseTransformer):
    """
    Removes columns that should not be predictive
    """

    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop

    def transform(self, x):
        return x.drop(self.columns_to_drop, axis="columns")

    def transform_explanation_shap(self, explanation):
        for col in self.columns_to_drop:
            explanation[col] = 0
        return explanation


class MultiTypeImputer(BaseTransformer):
    """
    Imputes, choosing a strategy based on column type.
    """

    def __init__(self):
        self.numeric_cols = None
        self.categorical_cols = None
        self.numeric_imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
        self.categorical_imputer = SimpleImputer(missing_values=np.nan, strategy="most_frequent")

    def fit(self, x):
        self.numeric_cols = x.select_dtypes(include="number").columns
        self.categorical_cols = x.select_dtypes(exclude="number").columns
        self.numeric_imputer.fit(x[self.numeric_cols])
        self.categorical_imputer.fit(x[self.categorical_cols])

    def transform(self, x):
        new_numeric_cols = self.numeric_imputer.transform(x[self.numeric_cols])
        new_categorical_cols = self.categorical_imputer.transform(x[self.categorical_cols])
        return pd.concat([pd.DataFrame(new_numeric_cols, columns=self.numeric_cols, index=x.index),
                          pd.DataFrame(new_categorical_cols, columns=self.categorical_cols,
                                       index=x.index)], axis=1)
