import logging
import re

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder

from pyreal.transformers import BreakingTransformError, Transformer

log = logging.getLogger(__name__)


def _helper_transform_explanation_additive_with_mappings(wrapped_explanation, mappings):
    if wrapped_explanation.ndim == 1:
        wrapped_explanation = wrapped_explanation.reshape(1, -1)
    for original_feature in mappings.categorical_to_one_hot.keys():
        encoded_features = mappings.categorical_to_one_hot[original_feature].keys()
        summed_contribution = wrapped_explanation[encoded_features].sum(axis=1)
        wrapped_explanation = wrapped_explanation.drop(encoded_features, axis="columns")
        wrapped_explanation[original_feature] = summed_contribution
    return wrapped_explanation


def _generate_one_hot_to_categorical(categorical_to_one_hot):
    one_hot_to_categorical = {}
    for cf in categorical_to_one_hot:
        for ohf in categorical_to_one_hot[cf]:
            one_hot_to_categorical[ohf] = (cf, categorical_to_one_hot[cf][ohf])

    return one_hot_to_categorical


def _generate_categorical_to_one_hot(one_hot_to_categorical):
    categorical_to_one_hot = {}
    for ohf in one_hot_to_categorical:
        cf = one_hot_to_categorical[ohf][0]
        value = one_hot_to_categorical[ohf][1]
        if cf not in categorical_to_one_hot:
            categorical_to_one_hot[cf] = {ohf: value}
        else:
            categorical_to_one_hot[cf][ohf] = value
    return categorical_to_one_hot


def _generate_from_df(df):
    categorical_to_one_hot = {}
    for i in range(df.shape[0]):
        cf = df["categorical"][i]
        ohf = df["one_hot_encoded"][i]
        value = df["value"][i]
        if cf not in categorical_to_one_hot:
            categorical_to_one_hot[cf] = {ohf: value}
        else:
            categorical_to_one_hot[cf][ohf] = value
    return categorical_to_one_hot


class Mappings:
    def __init__(self, categorical_to_one_hot, one_hot_to_categorical):
        """
        Initialize a new mappings object
        For common use, use Mappings.generate_mapping()

        Args:
            categorical_to_one_hot (dictionary):
                {categorical_feature_name : {OHE_feature_name : value, ...}, ... }
            one_hot_to_categorical (dictionary):
                {OHE_feature_name : (categorical_feature_name, value), ...}
        """

        self.categorical_to_one_hot = categorical_to_one_hot
        self.one_hot_to_categorical = one_hot_to_categorical

    @staticmethod
    def generate_mappings(
        categorical_to_one_hot=None, one_hot_to_categorical=None, dataframe=None
    ):
        """
        Generate a new Mappings object using one of the input formats
        All but one keyword should be None

        Args:
            categorical_to_one_hot:
                {categorical_feature_name : {OHE_feature_name : value, ...}, ... }
            one_hot_to_categorical:
                {OHE_feature_name : (categorical_feature_name, value), ...}
            dataframe:
                DataFrame with three columns named [categorical, one_hot_encoded, values]
                ie., [["A_a", "A", "a"], ["A_b", "B", "b"]]
        Returns:
            Mappings
                A Mappings objects representing the column relationships
        """

        if categorical_to_one_hot is not None:
            return Mappings(
                categorical_to_one_hot, _generate_one_hot_to_categorical(categorical_to_one_hot)
            )
        if one_hot_to_categorical is not None:
            return Mappings(
                _generate_categorical_to_one_hot(one_hot_to_categorical), one_hot_to_categorical
            )
        if dataframe is not None:
            categorical_to_one_hot = _generate_from_df(dataframe)
            return Mappings(
                categorical_to_one_hot, _generate_one_hot_to_categorical(categorical_to_one_hot)
            )


class OneHotEncoder(Transformer):
    """
    One-hot encodes categorical feature values
    """

    def __init__(self, columns=None, **kwargs):
        """
        Initializes the base one-hot encoder

        Args:
            columns (dataframe column label type or list of dataframe column label type):
                Label of column to select, or an ordered list of column labels to select
        """
        self.ohe = SklearnOneHotEncoder(sparse_output=False).set_output(transform="pandas")
        if columns is not None and not isinstance(columns, (list, tuple, np.ndarray, pd.Index)):
            columns = [columns]
        self.columns = columns
        self.orig_columns = None
        self.onehot_columns = None
        super().__init__(**kwargs)

    def fit(self, x, **params):
        """
        Fit this transformer to data

        Args:
            x (DataFrame of shape (n_instances, n_features)):
                The dataset to fit to

        Returns:
            None
        """

        if self.columns is None:
            self.columns = x.columns
        self.orig_columns = x.columns
        self.ohe.fit(x[self.columns])
        self.onehot_columns = self.ohe.get_feature_names_out(self.columns)
        return super().fit(x)

    def data_transform(self, x):
        """
        One-hot encode `x`.
        Args:
            x (DataFrame of shape (n_instances, n_features)):
                The dataset to transform

        Returns:
            DataFrame of shape (n_instances, n_transformed_features):
                The one-hot encoded dataset
        """

        if not self.fitted:
            raise RuntimeError("Must fit one hot encoder before transforming")
        x_to_encode = x[self.columns]
        x_cat_ohe = self.ohe.transform(x_to_encode)
        return pd.concat([x.drop(self.columns, axis="columns"), x_cat_ohe], axis=1)

    def inverse_data_transform(self, x_new):
        """
        Transforms one-hot encoded data `x_new` back into the original feature space.
        Args:
            x_new (DataFrame of shape (n_instances, n_transformed_features)):
                The one-hot encoded dataset

        Returns:
            DataFrame of shape (n_instances, n_features):
                The inverse-transformed dataset
        """
        if not self.fitted:
            raise RuntimeError("Must fit one hot encoder before performing inverse transform")
        index = x_new.index
        x_orig = self.ohe.inverse_transform(x_new[self.onehot_columns])
        x_orig = pd.DataFrame(x_orig, columns=self.columns, index=index)
        unordered_x = pd.concat([x_new.drop(self.onehot_columns, axis="columns"), x_orig], axis=1)

        return unordered_x[self.orig_columns]

    def inverse_transform_explanation_additive_feature_contribution(self, explanation):
        """
        Combine the contributions of one-hot-encoded features through adding to get the
        contributions of the original categorical feature.

        Args:
            explanation (AdditiveFeatureContributionExplanation):
                The explanation to transform

        Returns:
            AdditiveFeatureContributionExplanation:
                The transformed explanation
        """
        return explanation.update_explanation(self._helper_summed_values(explanation.get()))

    def inverse_transform_explanation_additive_feature_importance(self, explanation):
        """
        Combine the importances of one-hot-encoded features through adding to get the
        contributions of the original categorical feature.

        Args:
            explanation (AdditiveFeatureImportanceExplanation):
                The explanation to transform

        Returns:
            AdditiveFeatureImportanceExplanation:
                The transformed explanation
        """
        return explanation.update_explanation(self._helper_summed_values(explanation.get()))

    def inverse_transform_explanation_feature_based(self, explanation):
        """
        For non-additive feature-based explanations, the contributions or importances of
        the one-hot encoded features cannot be combined. This will result in a different feature
        space in the explanation than the pre-transformed data. Therefore, attempting to reverse
        the transform on the explanation in this case should stop the explanation transform
        process.

        Args:
            explanation (FeatureBased):
                The explanation to transform

        Raises:
            BreakingTransformError
        """
        raise BreakingTransformError

    def transform_explanation_feature_based(self, explanation):
        """
        For feature-based explanations, the contributions or importances of categorical features
        cannot be split into per-category features. This will result in a different feature
        space in the explanation than the pre-transformed data. Therefore, attempting to one-hot
        encode an explanation will should the explanation transform process.

        If you'd like to get your explanation one-hot encoded, this procedure should be applied
        to the data before generating the explanation if possible.

        Args:
            explanation (FeatureBased):
                The explanation to transform

        Raises:
            BreakingTransformError
        """
        log.info(
            "Explanation cannot be one-hot encoded with the available information. "
            "If you'd like to get your explanation one-hot encoded, "
            "this procedure should be applied to the data before generating "
            "the explanation if possible."
        )
        raise BreakingTransformError

    def transform_explanation_decision_tree(self, explanation):
        """
        Features cannot be added to encoded in existing decision trees,
        so raise a BreakingTransformError

        Args:
            explanation (DecisionTree):
                The explanation to be transformed

        Raises:
            BreakingTransformError

        """
        raise BreakingTransformError

    def inverse_transform_explanation_decision_tree(self, explanation):
        """
        Features cannot be decoded in existing decision trees, so raise a BreakingTransformError

        Args:
            explanation (DecisionTree):
                The explanation to be transformed

        Raises:
            BreakingTransformError

        """
        raise BreakingTransformError

    def _helper_summed_values(self, explanation):
        """
        Sum together the items in the explanation.
        Args:
            explanation: a list of values, one per feature

        Returns:
            the values summed together for all features involved in the one-hot encoding
        """
        if explanation.ndim == 1:
            explanation = explanation.reshape(1, -1)
        encoded_columns = self.ohe.get_feature_names_out(self.columns)
        longest_first = sorted(self.orig_columns, key=len, reverse=True)
        regex = r"(" + "|".join(map(re.escape, longest_first)) + ")"
        summed_contribution = (
            explanation[encoded_columns]
            .groupby(
                explanation[encoded_columns].columns.str.extract(regex, expand=False),
                axis=1,
            )
            .sum()
        )
        explanation = explanation.drop(columns=encoded_columns)
        explanation = pd.concat([explanation, summed_contribution], axis=1)
        return explanation


class MappingsOneHotEncoder(Transformer):
    """
    Converts data from categorical form to one-hot-encoded, with feature names based on a
    mappings object which includes two dictionaries
    """

    def __init__(self, mappings, **kwargs):
        """
        Initialize the transformer

        Args:
            mappings (Mappings):
                Mappings from categorical column names to one-hot-encoded
        """
        self.mappings = mappings
        super().__init__(**kwargs)

    def data_transform(self, x):
        """
        One-hot encode `x`.
        Args:
            x (DataFrame of shape (n_instances, n_features)):
                The dataset to transform

        Returns:
            DataFrame of shape (n_instances, n_transformed_features):
                The one-hot encoded dataset
        """
        cols = x.columns
        num_rows = x.shape[0]
        ohe_data = {}
        for col in cols:
            values = x[col]
            if col not in self.mappings.categorical_to_one_hot:
                ohe_data[col] = values
            else:
                ohe_feature_dict = self.mappings.categorical_to_one_hot[col]
                for ohe_feature in ohe_feature_dict:
                    ohe_data[ohe_feature] = np.zeros(num_rows, dtype=bool)
                    ohe_data[ohe_feature][np.where(values == ohe_feature_dict[ohe_feature])] = True
        return pd.DataFrame(ohe_data)

    def inverse_data_transform(self, x_onehot):
        """
        Transforms one-hot encoded data `x_onehot` back into categorical form.
        Args:
            x_onehot (DataFrame of shape (n_instances, n_transformed_features)):
                The one-hot encoded dataset

        Returns:
            DataFrame of shape (n_instances, n_features):
                The dataset in categorical form
        """
        cat_data = {}
        cols = x_onehot.columns
        num_rows = x_onehot.shape[0]

        for col in cols:
            if col not in self.mappings.one_hot_to_categorical:
                cat_data[col] = x_onehot[col]
            else:
                new_name = self.mappings.one_hot_to_categorical[col][0]
                if new_name not in cat_data:
                    cat_data[new_name] = np.empty(num_rows, dtype="object")
                # TODO: add functionality to handle defaults
                cat_data[new_name][np.where(x_onehot[col] == 1)] = (
                    self.mappings.one_hot_to_categorical[col][1]
                )
        return pd.DataFrame(cat_data)

    def inverse_transform_explanation_additive_feature_contribution(self, explanation):
        return explanation.update_explanation(
            _helper_transform_explanation_additive_with_mappings(explanation.get(), self.mappings)
        )

    def inverse_transform_explanation_additive_feature_importance(self, explanation):
        return explanation.update_explanation(
            _helper_transform_explanation_additive_with_mappings(explanation.get(), self.mappings)
        )


class MappingsOneHotDecoder(Transformer):
    """
    Converts data from one-hot encoded form to categorical, with feature names based on a
    mappings object which includes two dictionaries
    """

    def __init__(self, mappings, **kwargs):
        """
        Initialize the transformer

        Args:
            mappings (Mappings):
                Mappings from categorical column names to one-hot-encoded
        """
        self.mappings = mappings
        super().__init__(**kwargs)

    def data_transform(self, x):
        """
        One-hot decode `x`.
        Args:
            x (DataFrame of shape (n_instances, n_features)):
                The dataset to transform

        Returns:
            DataFrame of shape (n_instances, n_transformed_features):
                The one-hot decoded dataset
        """
        cat_data = {}
        cols = x.columns
        num_rows = x.shape[0]

        for col in cols:
            if col not in self.mappings.one_hot_to_categorical:
                cat_data[col] = x[col]
            else:
                new_name = self.mappings.one_hot_to_categorical[col][0]
                if new_name not in cat_data:
                    cat_data[new_name] = np.empty(num_rows, dtype="object")
                # TODO: add functionality to handle defaults
                cat_data[new_name][np.where(x[col] == 1)] = self.mappings.one_hot_to_categorical[
                    col
                ][1]
        return pd.DataFrame(cat_data)

    def inverse_data_transform(self, x_cat):
        """
        Transforms one-hot decoded data `x_cat` back into one-hot encoded form.
        Args:
            x_cat (DataFrame of shape (n_instances, n_transformed_features)):
                The dataset to transform

        Returns:
            DataFrame of shape (n_instances, n_features):
                The one-hot encoded dataset
        """
        cols = x_cat.columns
        num_rows = x_cat.shape[0]
        ohe_data = {}
        for col in cols:
            values = x_cat[col]
            if col not in self.mappings.categorical_to_one_hot:
                ohe_data[col] = values
            else:
                ohe_feature_dict = self.mappings.categorical_to_one_hot[col]
                for ohe_feature in ohe_feature_dict:
                    ohe_data[ohe_feature] = np.zeros(num_rows, dtype=bool)
                    ohe_data[ohe_feature][np.where(values == ohe_feature_dict[ohe_feature])] = True
        return pd.DataFrame(ohe_data)

    def transform_explanation_additive_feature_contribution(self, explanation):
        """
        Transforms additive contribution explanations

        Args:
            explanation (AdditiveFeatureContributionExplanation):
                The explanation to be transformed

        Returns:
            AdditiveFeatureContributionExplanation:
                The transformed explanation
        """
        return explanation.update_explanation(
            _helper_transform_explanation_additive_with_mappings(explanation.get(), self.mappings)
        )

    def transform_explanation_additive_feature_importance(self, explanation):
        """
        Transforms additive importance explanations

        Args:
            explanation (AdditiveFeatureImportanceExplanation):
                The explanation to be transformed

        Returns:
            AdditiveFeatureImportanceExplanation:
                The transformed explanation
        """
        return explanation.update_explanation(
            _helper_transform_explanation_additive_with_mappings(explanation.get(), self.mappings)
        )
