from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class BaseTransformer(ABC):
    @abstractmethod
    def transform(self, data):
        pass


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

