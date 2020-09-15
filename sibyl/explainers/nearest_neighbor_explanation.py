# includes code from
# https://towardsdatascience.com/k-medoids-clustering-on-iris-data-set-1931bf781e05

from sklearn.neighbors import NearestNeighbors
import numpy as np
import pickle


def fit_nearest_neighbor(X, metric='minkowski',
                         savefile=None, return_result=False):
    """
    Fit the nearest neighbor algorithm to the dataset.

    :param X: array_like of shape (n_samples, n_features)
              The dataset to search for neighbors in
    :param savefile: file object
          Where the save the neighbors list. If None, don't save
    :param return_result: boolean
          If true, return the resulting neighbors, else return none
    :param metric: distance metric to use
    :return: None
    """
    nbrs = NearestNeighbors(n_neighbors=1,
                            algorithm='ball_tree',
                            metric=metric).fit(X)
    if savefile is not None:
        pickle.dump(nbrs, savefile)
    if return_result:
        return nbrs


def load_nearest_neighbor(file):
    """
    Load a nearest neighbor fit.

    :param file: file object
           The file of the pickled explainer
    :return: the nearest neighbor object
    """
    return pickle.load(file)


def nearest_neighbor(nbrs, x, N=1, desired_y=None, y=None,
                     search_by=10, search_depth=1000):
    """
    Find the most similar input to x in the neighbors list.
    Requires fit_nearest_neighbor to be called first

    :param nbrs: the result of fit_nearest_neighbors
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
    :except ValueError
            If desired_y is not None, but y is None and no y was provided
            at fitting time
    """
    if desired_y is None:
        nbr_inds = nbrs.kneighbors(
            x, return_distance=False, n_neighbors=N)[0]
        return nbr_inds

    if desired_y is not None and y is None:
        raise ValueError("Must provide true values when requesting a y if "
                         "they were not provided when fitting")
    if desired_y is not None:
        y = np.asanyarray(y)
        # TODO: this function would be much cleaner if we knew how many samples
        #  nbrs had, but NearestNeighbors does not share this
        current_search_N = search_by
        inds = []
        while True:
            try:
                nbr_inds = nbrs.kneighbors(x, return_distance=False,
                                       n_neighbors=current_search_N)[0]
            except ValueError:
                print("Couldn't find enough appropriate " +
                      "neighbors within search_depth")
                return inds
            else:
                for ind in nbr_inds:
                    if ind in inds:
                        continue
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
