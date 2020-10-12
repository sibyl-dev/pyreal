import numpy as np
from utils.utils import identity_transform


def binary_flip_all(predict, x, features=None, transform=identity_transform):
    """
    Get the results of flipping each binary variable in an input

    :param predict: function,
                    array_like of size (n_samples, n_features) -> (n_samples,)
           The prediction function to use
    :param x: array_like of size (n_features,)
           The input to explain. Should be unstandardized
    :param features: array_like
            Numeric indices of the features to swap
            If none, flip all features
    :param transform: function, array_like of size (n_features,)
                                        -> array_like of size (n_features)
           The transformation function to prepare inputs for predict
    :return: pair of lists of length len(features), n_features if features=None
             The first list is the resulting prediction from flipping each
             binary variable, and the second list is the new values after
             each flip.
    """
    flip_preds = []
    values = []
    x = np.asanyarray(x)
    if features is None:
        features = np.arange(len(x))
    for feat in features:
        mod_x = x.copy()
        mod_x[feat] = 1 - x[feat]
        value = 1 - x[feat]
        mod_x = transform(mod_x)
        pred = predict(mod_x.reshape(1, -1))
        flip_preds.append(pred[0])
        values.append(value)
    return flip_preds, values


def modify_and_repredict(predict, X, features, new_values,
                         transform=identity_transform):
    """
    Make changes to x and then return the new prediction

    :param predict: function,
                    array_like of size (n_samples, n_features) -> (n_samples,)
           The prediction function to use
    :param X: array_like of shape (n_samples, n_features), or shape (n_features,)
              for a single input.
              The untransformed data to modify
    :param features: array_like
           Numeric indices of the features to change
    :param new_values: array_like, same length as features
                       len(new_values)[n] = n_samples or new_values[n] = scalar
                       for single inputs.
           The new values to give the features
    :param transform: function, array_like of size (n_samples, n_features)
                                -> array_like of size (n_samples, n_features)
           The transformation function to prepare inputs for predict
    :return: array_like of length (n_samples, )
            The new predictions after making the requested changes
    :except ValueError
            If features and new_values are not the same length.
    """
    if len(features) != len(new_values):
        raise ValueError("features and new_values must be the same length")

    X = np.asanyarray(X)
    new_values = np.asanyarray(new_values)
    if len(features) > 0 and X.ndim != new_values.ndim:
        raise ValueError("X and new_values must have same dimensionality")

    if X.ndim == 1:
        X = X.reshape(1, -1)
    X_new = X.copy()
    for i, feature in enumerate(features):
        if new_values[i].size != X.shape[0]:
            raise ValueError("Invalid number of values for number of samples")
        X_new[:,feature] = new_values[i]
    X_new = transform(X_new)
    new_pred = predict(X_new)
    return new_pred
