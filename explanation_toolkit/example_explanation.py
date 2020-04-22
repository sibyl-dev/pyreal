# TODO: Under construction, needs refactoring
from sklearn.neighbors import NearestNeighbors

nbrs = None
saved_y = None


def fit_nearest_neighbor(X, y=None):
    """
    Fit the nearest neighbor algorithm to the dataset.

    :param X: the dataset to search
    :param y: the true values that accompany X
    :return: None
    """
    global nbrs
    global saved_y
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(X)
    saved_y = y


def nearest_neighbor(x, N=1, desired_y=None, ys=None,
                     search_by=10, search_depth=1000):
    """
    Find the most similar input to x in the neighbors list.
    Requires fit_nearest_neighbor to be called first

    :param x: the input to search for similar inputs to
    :param N: The number of neighbors to return
    :param desired_y: Only return neighbors with this y
    :param ys: ys associated with the dataset
    :param search_by: Number of neighbors to look at at a time when searching
                      for a desired y
    :param search_depth: The maximum number of neighbors to look at before
                         returning
    :return: the index in X (from fit_nearest_neighbor) of the nearest input
    """
    assert(nbrs is not None,
           "Must call fit_nearest_neighbor before calling nearest_neighbor")

    if desired_y is None:
        nbr_inds = nbrs.kneighbors(x, return_distance=False, n_neighbors=N)[0]
        return nbr_inds

    if desired_y is not None and saved_y is None and ys is None:
        raise ValueError("Must provide true values when requesting a y if "
                         "they were not provided when fitting")
    if desired_y is not None:
        current_search_N = search_by
        inds = []
        while True:
            nbr_inds = nbrs.kneighbors(x, return_distance=False,
                                       n_neighbors=current_search_N)[0]
            for ind in nbr_inds:
                y = ys.iloc[ind]
                if y == desired_y:
                    inds.append(ind)
                    if len(inds) >= N:
                        return inds
            current_search_N += search_by
            if current_search_N > search_depth:
                print("Couldn't find enough appropriate " +
                      "neighbors within search_depth")
                return inds
