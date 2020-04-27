"""
Predefined templates for distance functions
"""
from sklearn.neighbors import DistanceMetric
import sklearn.metrics
from abc import ABC, abstractmethod
from scipy.spatial.distance import cdist

def minkowski(x, y):
    return sklearn.metrics.pairwise_distances(x.reshape(1,-1), y.reshape(1,-1), metric='minkowski')


class DistanceTemplate(ABC):
    """
    Basic template for predefined distance methods with predefined conditions
    """
    @abstractmethod
    def distance(self, x, y):
        pass


class PartialFeatureDistance(DistanceTemplate):
    """
    Allows for distances between inputs while only considering a subset of
    features
    """
    def __init__(self, features_to_use, base_distance=minkowski):
        """

        :param features_to_use: array of ints
               Indexes of the feature to use in computing distance
        :param base_distance: callable
               The distance function to use
        """
        self.features_to_use = features_to_use
        self.base_distance = base_distance

    def distance(self, x, y):
        x_to_use = x[self.features_to_use]
        y_to_use = y[self.features_to_use]

        return self.base_distance(x_to_use, y_to_use)


class WeightedEuclidean(DistanceTemplate):
    def __init__(self, weights):
        """
        Implements weighted euclidean
        :param weights: the weights for each feature, in order
        """
        self.weights = weights

    def distance(self, x, y):
        return cdist(x, y, metric='euclidean', w=self.weights)
