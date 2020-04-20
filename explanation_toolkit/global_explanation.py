from eli5.sklearn import PermutationImportance
import numpy as np


def get_global_importance(model, X_train, y_train):
    """
    Get the overall importances for all features in x

    :param model: the model to explain
    :param X_train: the standardized training set to calculate the contributions
    :param y_train: the true values of the items in X_train, in the format returned by the model
    :return: the importance of each feature in X_train
    """
    perm = PermutationImportance(model, random_state=1, scoring="neg_mean_squared_error").fit(X_train,y_train)
    importances = perm.feature_importances_
    return importances


def get_xs_with_predictions(output, predict, X):
    """
    Return an dataframe of rows in X that get predicted as output

    :param output: the output to look for
    :param predict: the prediction function
    :param X: the dataframe of data to look for
    :return: a dataframe including the rows of X that result in X when run
             through predict
    """
    preds_train = predict(X)
    xs_of_interest = X[preds_train == output]
    return xs_of_interest


def summary_count(X):
    """
    Count the number of each value that appears for each feature (column) in X
    :param X: A dataframe
    :return: A dictionary of feature name -> two lists, [unique values, counts]
    """
    all_counts = {}
    for feature in X.columns:
        values = X[feature]
        unique, counts = np.unique(values, return_counts=True)
        all_counts[feature] = [unique, counts]
    return all_counts


def summary_metrics(X):
    """
    Find the minimum, 1st quartile, median, 2nd quartile, and maximum of the
    values for each feature in X
    :param X: A dataframe
    :return: A dictionary of feature name ->
    [minimum, 1st quartile, median, 2nd quartile, and maximum]
    """
    all_metrics = {}
    for feature in X.columns:
        values = X[feature]
        quartiles = values.quantile([0.25, 0.5, 0.75])
        maximum = values.max()
        minimum = values.min()
        all_metrics[feature] = [minimum, quartiles.iloc[0], quartiles.iloc[1],
                                quartiles.iloc[2], maximum]
    return all_metrics


def overview_categorical(output, predict, X, features=None):
    """
    Return the number of each feature value for elements in X that are
    predicted as output

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


def overview_metrics(output, predict, X, features=None):
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
    return summary_metrics(xs_of_interest)

