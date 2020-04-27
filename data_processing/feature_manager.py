"""
Contains functionality to improve feature readability/interpretability

[Set of original features + set of original values] -> [Set of new features + set of new values]
Notes:
    1. Size of these two sets may be different
    2. Values may or may not matter

Possible conversions:
[Original feature name + value] -> [Readable feature name + value]
[Several binary features + Trues/Falses] -> [One feature + category]
"""
from abc import ABC,abstractmethod

# TODO: Draft, may be refactored or removed


class FeatureManager(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def parse_features(self, features, feature_types):
        """
        Parses a set of a features and returns a updated set of features
        :param features: the features to update
        :param feature_types: the corresponding feature types
        :return: updated features, feature_types
        """
        pass

    @abstractmethod
    def parse_values(self, X):
        """
        Parses specific values and applies local effects
        :param X: a set of inputs
        :return: An updated version of X with local effects applied
        """
        pass


class ReadableNames(FeatureManager):
    def __init__(self, feature_dict=None):
        super().__init__()
        if feature_dict is None:
            self.features_to_readable = {}
        else:
            self.features_to_readable = feature_dict

    def add_feature(self, name, readable):
        self.features_to_readable[name] = readable

    def parse_features(self, features, feature_type=None):
        super().parse_features(features, feature_type)
        readables = []
        for feature in features:
            if feature in self.features_to_readable:
                readables.append(self.features_to_readable[feature])
            else:
                readables.append(feature)
        return features, feature_type

    def parse_values(self, X):
        super().parse_values(X)
        return X


class BooleanToCatergorical(FeatureManager):
    def ___init__(self, mappings):
        self.original_to_ = mappings

    def add_mapping(self, new_feature, original_features):
        pass


