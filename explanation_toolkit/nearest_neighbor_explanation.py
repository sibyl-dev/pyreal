from sklearn.neighbors import NearestNeighbors
import numpy as np


class NearestNeighborExplanation:
    def __init__(self):
        self.nbrs = None
        self.n_samples = 0
        self.y = None

    def fit_nearest_neighbor(self, X, y=None, metric='minkowski'):
        """
        Fit the nearest neighbor algorithm to the dataset.

        :param X: array_like of shape (n_samples, n_features)
                  The dataset to search for neighbors in
        :param y: array_like of shape (n_samples, )
                  The true values that accompany X
        :param metric: distance metric to use
        :return: None
        """
        self.nbrs = NearestNeighbors(n_neighbors=1,
                                     algorithm='ball_tree',
                                     metric=metric).fit(X)
        self.n_samples = np.asanyarray(X).shape[0]
        if y is not None:
            self.y = np.asanyarray(y)

    def nearest_neighbor(self, x, N=1, desired_y=None, y=None,
                         search_by=10, search_depth=1000):
        """
        Find the most similar input to x in the neighbors list.
        Requires fit_nearest_neighbor to be called first

        :param x: array_like of shape (n_features, )
               The input to search for similar inputs to
        :param N: integer
               The number of neighbors to return
        :param desired_y: object
               Only return neighbors with this y
        :param y: array_like of size (n_samples, )
               True values associated with the dataset, not required if y was
               provided at fitting or desired_y is None
        :param search_by: integer
               Number of neighbors to look at at a time when searching
               for a desired y
        :param search_depth: integer
               The maximum number of neighbors to look at before returning
        :return: integer
                 The row index in X (from fit_nearest_neighbor) of the nearest
                 input
        :except AssertError
                If fit_nearest_neighbor has not been called yet
        :except ValueError
                If desired_y is not None, but y is None and no y was provided
                at fitting time
        """
        assert self.nbrs is not None,\
            "Must call fit_nearest_neighbor before calling nearest_neighbor"

        if desired_y is None:
            nbr_inds = self.nbrs.kneighbors(
                x, return_distance=False, n_neighbors=N)[0]
            return nbr_inds

        if desired_y is not None and self.y is None and y is None:
            raise ValueError("Must provide true values when requesting a y if "
                             "they were not provided when fitting")
        if desired_y is not None:
            if y is None:
                y = self.y
            else:
                y = np.asanyarray(y)
            current_search_N = min(search_by, self.n_samples)
            inds = []
            while True:
                nbr_inds = self.nbrs.kneighbors(x, return_distance=False,
                                                n_neighbors=current_search_N)[0]
                for ind in nbr_inds:
                    this_y = y[ind]
                    if this_y == desired_y:
                        inds.append(ind)
                        if len(inds) >= N:
                            return inds
                current_search_N += search_by
                if current_search_N > search_depth:
                    print("Couldn't find enough appropriate " +
                          "neighbors within search_depth")
                    return inds
                if current_search_N > self.n_samples:
                    print("Couldn't find enough appropriate " +
                          "neighbors within training set")
                    return inds
