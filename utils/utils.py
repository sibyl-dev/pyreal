import numpy as np


def identity_transform(x):
    """
    Identity transform x (for when no transformation is necessary for model)

    :param x: the input to transform
    :return: x
    """
    return x


def get_inds(names, all_names):
    """
    Helper to get the indices from feature names
    :param names: array_like or scalar
           The name(s) of the features to get indices for
    :param all_names: array_like
           The ordered list of features to take from
    :return: array_like of integers or integer
            A list of indices (or single integer if a scalar was input)
    """
    all_names = np.asanyarray(all_names)
    sorter = np.argsort(all_names)
    return sorter[np.searchsorted(all_names, names, sorter=sorter)]


class IndsGetter:
    """
    Wrapper to reuse column order
    """
    def __init__(self, all_names):
        """
        :param all_names: array_like
           The ordered list of features to take from
        """
        self.all_names = all_names
        self.sorter = np.argsort(all_names)

    def get_inds(self, names):
        """
        :param names: array_like or scalar
                The name(s) of the features to get indices for
        :return: array_like of integers or integer
                 A list of indices (or single integer if a scalar was input)
        """
        return self.sorter[
            np.searchsorted(self.all_names, names, sorter=self.sorter)]
