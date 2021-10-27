import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder

from pyreal.transformers import BaseTransformer


def generate_one_hot_to_categorical(categorical_to_one_hot):
    one_hot_to_categorical = {}
    for cf in categorical_to_one_hot:
        for ohf in categorical_to_one_hot[cf]:
            one_hot_to_categorical[ohf] = (cf, categorical_to_one_hot[cf][ohf])

    return one_hot_to_categorical


def generate_categorical_to_one_hot(one_hot_to_categorical):
    categorical_to_one_hot = {}
    for ohf in one_hot_to_categorical:
        cf = one_hot_to_categorical[ohf][0]
        value = one_hot_to_categorical[ohf][1]
        if cf not in categorical_to_one_hot:
            categorical_to_one_hot[cf] = {ohf: value}
        else:
            categorical_to_one_hot[cf][ohf] = value
    return categorical_to_one_hot


def generate_from_df(df):
    # TODO: rename columns to be more natural
    categorical_to_one_hot = {}
    for i in range(df.shape[0]):
        cf = df["name"][i]
        ohf = df["original_name"][i]
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

        :param categorical_to_one_hot: dictionary
               {categorical_feature_name : {OHE_feature_name : value, ...}, ... }
        :param one_hot_to_categorical: dictionary
               {OHE_feature_name : (categorical_feature_name, value), ...}
        """
        self.categorical_to_one_hot = categorical_to_one_hot
        self.one_hot_to_categorical = one_hot_to_categorical

    @staticmethod
    def generate_mappings(categorical_to_one_hot=None,
                          one_hot_to_categorical=None,
                          dataframe=None):
        """
        Generate a new Mappings object using one of the input formats
        All but one keyword should be None

        :param categorical_to_one_hot: dictionary
               {categorical_feature_name : {OHE_feature_name : value, ...}, ... }
        :param one_hot_to_categorical:
               {OHE_feature_name : (categorical_feature_name, value), ...}
        :param dataframe:
               DataFrame # TODO: specify type
        :return:
        """
        if categorical_to_one_hot is not None:
            return Mappings(categorical_to_one_hot,
                            generate_one_hot_to_categorical(categorical_to_one_hot))
        if one_hot_to_categorical is not None:
            return Mappings(generate_categorical_to_one_hot(one_hot_to_categorical),
                            one_hot_to_categorical)
        if dataframe is not None:
            categorical_to_one_hot = generate_from_df(dataframe)
            return Mappings(categorical_to_one_hot,
                            generate_one_hot_to_categorical(categorical_to_one_hot))


class OneHotEncoder(BaseTransformer):
    def __init__(self, columns=None):
        self.ohe = SklearnOneHotEncoder(sparse=False)
        self.columns = columns
        self.is_fit = False

    def fit(self, x_orig):
        if self.columns is None:
            self.columns = x_orig.columns
        self.ohe.fit(x_orig[self.columns])
        self.is_fit = True

    def transform(self, x_orig):
        if not self.is_fit:
            raise RuntimeError("Must fit one hot encoder before transforming")
        x_to_encode = x_orig[self.columns]
        columns = self.ohe.get_feature_names(x_to_encode.columns)
        index = x_to_encode.index
        x_cat_ohe = self.ohe.transform(x_to_encode)
        x_cat_ohe = pd.DataFrame(x_cat_ohe, columns=columns, index=index)
        return pd.concat([x_orig.drop(self.columns, axis="columns"), x_cat_ohe], axis=1)

    def transform_explanation_shap(self, explanation):
        return self.helper_summed_values(explanation)

    # TODO: replace this with a more theoretically grounded approach to combining feature
    #  importance
    def transform_explanation_permutation_importance(self, explanation):
        return self.helper_summed_values(explanation)

    def helper_summed_values(self, explanation):
        """
        Sum together the items in the explanation.
        Args:
            explanation: a list of values, one per feature

        Returns:
            the values summed together for all features involved in the one-hot encoding
        """
        explanation = pd.DataFrame(explanation)
        if explanation.ndim == 1:
            explanation = explanation.reshape(1, -1)
        encoded_columns = self.ohe.get_feature_names(self.columns)
        for original_feature in self.columns:
            encoded_features = [item for item in encoded_columns if
                                item.startswith(original_feature + "_")]
            summed_contribution = explanation[encoded_features].sum(axis=1)
            explanation = explanation.drop(encoded_features, axis="columns")
            explanation[original_feature] = summed_contribution
        return explanation


class MappingsOneHotEncoder(BaseTransformer):
    """
    Converts data from categorical form to one-hot-encoded, with feature names based on a
    mappings object which includes two dictionaries
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


class MappingsOneHotDecoder(BaseTransformer):
    """
    Converts data from one-hot encoded form to categorical, with feature names based on a
    mappings object which includes two dictionaries
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
