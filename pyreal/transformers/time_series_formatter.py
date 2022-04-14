import numpy as np
import pandas as pd


def check_input(X, univariate=False, to_np=False, to_pd=False):
    """
    Validate input data.

    Args:
        X (DataFrame or ndarray):
            Input data
        univariate (bool):
            Set this to True if X is univariate.
        to_np (bool):
            If True, return X as a 3-dimensional NumPy array.
        to_pd (bool):
            If True, return X as a pandas DataFrame with MultiIndex-ed column.

    Returns:
        X : pd.DataFrame or np.ndarray
            Checked and possibly converted input data.
    """
    if to_np and to_pd:
        raise ValueError("`to_np` and `to_pd` cannot both be set to True")
    # check input type
    if not isinstance(X, (pd.DataFrame, np.ndarray)):
        raise ValueError(
            f"X must be a pd.DataFrame or a np.ndarray, but found: {type(X)}"
        )
    if isinstance(X, np.ndarray):
        if X.ndim == 2:
            X = X.reshape(X.shape[0], 1, X.shape[1])
        elif X.ndim == 1 or X.ndim > 3:
            raise ValueError(
                f"If passed as a np.ndarray, X must be a 2 or 3-dimensional "
                f"array, but found shape: {X.shape}"
            )
        if to_pd:
            X = numpy3d_to_df(X)
    # check univariate
    n_columns = X.shape[1]
    if univariate and n_columns > 1:
        raise ValueError(
            f"X must be univariate with X.shape[1] == 1, but found: "
            f"X.shape[1] == {n_columns}."
        )
    # check pd.DataFrame
    if isinstance(X, pd.DataFrame):
        if not is_valid_dataframe(X):
            raise ValueError(
                "If passed as a pd.DataFrame, X must be a MultiIndex pd.DataFrame."
            )
        # convert pd.DataFrame
        if to_np:
            X = df_to_numpy3d(X)

    return X


def is_valid_dataframe(X):
    """
    Check if the input is a valid DataFrame.

    Args:
        X: Input data
    Returns:
        bool: Whether the input is a MultiIndex DataFrame
    """
    if not isinstance(X, pd.DataFrame):
        return False
    elif not isinstance(X.columns, pd.MultiIndex):
        return False
    elif X.columns.nlevels != 2:
        return False
    return True


def numpy2d_to_df(X, var_name=None, timestamps=None):
    """
    Convert 2D DataFrame or NumPy array to MultiIndexed DataFrame.

    Args:
        X (DataFrame or ndarray of shape (n_instances, n_timepoints)):
            Input DataFrame or ndarray
        column_name (None or String):
            Optional name to use for the variable
        timestamps (None or list-like with length of n_timepoints):
            Time series index of transformed DataFrame

    Returns:
        DataFrame in MultiIndexed format
    """
    columns = None
    if isinstance(X, pd.DataFrame):
        columns = X.columns
        X = X.to_numpy()

    n_instances, n_timepoints = X.shape

    if timestamps is None and columns is None:
        timestamps = np.arange(n_timepoints)
    elif timestamps is None:
        timestamps = columns

    # create indices for MultiIndex DataFrame
    if var_name is None:
        mi = pd.MultiIndex.from_product([[0], timestamps])
    else:
        mi = pd.MultiIndex.from_product([[var_name], timestamps])
    df = pd.DataFrame(X, columns=mi)
    return df


def numpy3d_to_df(X, column_names=None, timestamps=None):
    """
    Convert NumPy ndarray with shape (n_instances, n_columns, n_timepoints) into
    DataFrame with MultiIndex)

    Args:
        X (ndarray with shape (n_instances, n_columns, n_timepoints)):
            3-dimensional NumPy array to convert to MultiIndex-ed DataFrame format
        column_names (None or list-like):
            Optional list of names to use for variables
        timestamps (None or list-like):
            Optional list of names to use for naming column of time

    Returns:
        DataFrame with MultiIndex as columns
    """
    # check input
    if not isinstance(X, np.ndarray):
        raise ValueError(
            f"X must be a 3-dimensional NumPy array, but found: {type(X)}"
        )
    if X.ndim != 3:
        raise ValueError(
            f"X must be a 3-dimensional NumPy array, but found shape: {X.shape}"
        )
    n_instances, n_columns, n_timepoints = X.shape

    if column_names is None:
        column_names = [f"var_{str(i)}" for i in range(n_columns)]
    else:
        if len(column_names) != n_columns:
            raise ValueError(
                f"Input 3d NumPy array has {n_columns} columns, "
                f"but only {len(column_names)} names supplied."
            )
    if timestamps is None:
        timestamps = np.arange(n_timepoints)
    else:
        if len(timestamps) != n_timepoints:
            raise ValueError(
                f"Input 3d NumPy array has {n_timepoints} timepoints, "
                f"but only {len(timestamps)} timestamps supplied."
            )
    # create indices for MultiIndex DataFrame
    mi = pd.MultiIndex.from_product([column_names, timestamps])
    flatten_data = X.reshape((n_instances, n_columns*n_timepoints))
    df = pd.DataFrame(flatten_data, columns=mi)
    return df


def df_to_numpy2d(X):
    """
    **This function should only be used on univariate series.**
    Convert MultiIndexed pandas DataFrame into NumPy ndarray with shape
    (n_instances, n_timepoints).
    Args:
        X (pd.DataFrame):
            Input data
    Returns:
        2-dimensional NumPy array
    """
    array = df_to_numpy3d(X)
    return array.squeeze(axis=1)


def df_to_numpy3d(X):
    """
    Convert MultiIndexed pandas DataFrame into NumPy ndarray with shape
    (n_instances, n_columns, n_timepoints).
    Args:
        X (pd.DataFrame):
            Input data
    Returns:
        3-dimensional NumPy array
    """
    if not is_valid_dataframe(X):
        raise ValueError("Input DataFrame is not a valid DataFrame")

    n_instances = X.index.size
    n_columns, n_timepoints = X.columns.levshape
    array = X.to_numpy().reshape((n_instances, n_columns, n_timepoints))

    return array
