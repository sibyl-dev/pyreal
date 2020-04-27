"""
Maintains interpretable feature types for visualization purposes.
"""
from abc import ABC, abstractmethod


class FeatureType(ABC):
    @abstractmethod
    def __init__(self):
        pass


class IntNumericFeature(FeatureType):
    def __init__(self, min, max):
        """
        Defines any numeric feature with integer values
        :param min: Integer
              The minimum reasonable value
        :param max: Integer
               The maximum reasonable value
        """
        super().__init__()
        self.max = max
        self.min = min


class FloatNumericFeature(FeatureType):
    def __init__(self, min, max):
        """
        Defines any numeric feature that may have float values
        :param min: Float
              The minimum reasonable value
        :param max: Float
               The maximum reasonable value
        """
        super().__init__()
        self.min = min
        self.max = max


class BooleanFeature(FeatureType):
    def __init__(self):
        """
        Defines boolean features
        """
        super().__init__()
        pass


class CategoricalFeature(FeatureType):
    def __init__(self, categories):
        """
        Defines categorical features
        :param categories: list
               The possible classes for this feature
        """
        super().__init__()
        self.categories = categories
