from eli5.sklearn import PermutationImportance
import numpy as np


def get_global_importance(model, X, y):
    """
    Get the overall importances for all features in x.
    Current only supports sklearn estimators

    :param model: sklearn estimator
           The model to explain
    :param X: array-like of shape (n_samples, n_features)
           The standardized training set to calculate the contributions
    :param y: array-like of shape (n_samples, )
           The true values of the items in X_train
    :return: array of floats of shape (n_features, )
           The importance of each feature in X_train
    """
    # TODO: Update this function to be model type agnostic
    perm = PermutationImportance(model, random_state=1,
                                 scoring="neg_mean_squared_error").fit(X, y)
    importances = perm.feature_importances_
    return importances


def consolidate_importances(importances, categorical_feature_sets,
                            algorithm="max"):
    """
    Calculate the overall importances for categorical features from the
    one-hot encoded importances.
    :param importances: array of floats of shape (n_features, )
           the importances of each feature
    :param categorical_feature_sets: list of lists of integer indices
           each set contains the indices of features in some categorical feature
           set
    :param algorithm: "max" or "mean"
    :return:
    """
    if algorithm not in ["max", "mean"]:
        raise ValueError("Unsupported algorithm %s" % algorithm)
    importances = np.asanyarray(importances)
    importance_for_sets = []
    for feature_set in categorical_feature_sets:
        importance_for_set = None
        if algorithm is "max":
            importance_for_set = np.max(importances[feature_set])
        if algorithm is "mean":
            importance_for_set = np.mean(importances[feature_set])
        importance_for_sets.append(importance_for_set)
    return importance_for_sets


def get_rows_by_output(output, predict, X, row_labels=None):
    """
    Return the indices of the rows in X that get predicted as output

    :param output: int or array-like
           The output or outputs to look for
    :param predict: function array-like (X.shape) -> (X.shape[0])
           The prediction function
    :param X: array-like of shape (n_samples, n_features)
           The data to look through
    :param row_labels: None or array_like of shape (n_samples,)
           If not None, return the row_labels of relevant rows instead of
           numerical indices
    :return: array_like
            The indices or row_labels of the rows of X that result in output
            when run through predict
    """
    X = np.asanyarray(X)
    preds_train = predict(X)
    if np.isscalar(output):
        output = [output]
    xs_of_interest = np.isin(preds_train, output)
    if row_labels is None:
        row_labels = np.arange(0, len(xs_of_interest))
    else:
        row_labels = np.asanyarray(row_labels)
    return row_labels[xs_of_interest]


def summary_categorical(X):
    """
    Returns the unique values and counts for each column in X
    :param X: array_like of shape (n_samples, n_features)
           The data to summarize
    :return values: list of length n_features of arrays
                    The unique values for each feature
    :return count: list of length n_features
                   The number of occurrences of each unique value
                   for each feature
    """
    all_values = []
    all_counts = []
    X = np.asanyarray(X)
    for col in X.T:
        values, counts = np.unique(col, return_counts=True)
        all_values.append(values)
        all_counts.append(counts)
    return all_values, all_counts


def summary_numeric(X):
    """
    Find the minimum, 1st quartile, median, 2nd quartile, and maximum of the
    values for each column in X
    :param X: array_like of shape (n_samples, n_features)
           The data to summarize
    :return: A list of length (n_features) of lists of length 5
             The metrics for each feature:
             [minimum, 1st quartile, median, 2nd quartile, and maximum]
    """
    all_metrics = []
    X = np.asanyarray(X)
    for col in X.T:
        quartiles = np.quantile(col, [0.25, 0.5, 0.75])
        maximum = col.max()
        minimum = col.min()
        all_metrics.append([minimum, quartiles[0], quartiles[1],
                            quartiles[2], maximum])
    return all_metrics


# TODO: Fix these functions and move them to come overarching function file
'''def overview_categorical(output, predict, X, features=None):
    """
    Return a categorical summary 

    :param output: the output to filter on
    :param predict: the prediction function to use
    :param X: the dataset to look though
    :param features: the features to use
    :return: the number of each feature value that occurs among elements in X
    predicted as output.
    """
    xs_of_interest = get_xs_with_predictions(output, predict, X)
    if features is not None:
        xs_of_interest = xs_of_interest[features]
    return summary_count(xs_of_interest)


def overview_numerical(output, predict, X, features=None):
    """
    Return the max, min, and quartiles of the rows in X that are predicted as
    output.

    :param output: the output to filter on
    :param predict: the prediction function to use
    :param X: the dataframe dataset to look through
    :param features: the features to use
    :return: the minimum, 1st quartile, median, 3rd quartile, and maximum of
    each feature value for rows in X that are predicted as output
    """
    xs_of_interest = get_xs_with_predictions(output, predict, X)
    if features is not None:
        xs_of_interest = xs_of_interest[features]
    return summary_metrics(xs_of_interest)'''

