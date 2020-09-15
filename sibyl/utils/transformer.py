from abc import ABC, abstractmethod


class BaseTransformer(ABC):
    @abstractmethod
    def transform(self, data):
        pass


class MappingsEncoderTransformer(BaseTransformer):
    def __init__(self, mappings):
        self.mappings = mappings

    def transform(self, data):
        pass


class MappingsDecoderTransformer(BaseTransformer):
    def __init__(self, mappings):
        self.mappings = mappings

    def transform(self, data):
        pass


class FeatureSelectTransformer(BaseTransformer):
    def __init__(self, feature_names):
        self.feature_names = feature_names

    def transform(self, data):
        return data[self.feature_names]

