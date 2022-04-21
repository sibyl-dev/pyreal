import numpy as np
import pandas as pd

from pyreal.transformers import Transformer


def _check_is_np2d(x):
    if not isinstance(x, np.ndarray):
        raise ValueError(
            f"Input data must be an np.ndarray, but found: {type(x)}"
        )
    if x.ndim != 2:
        raise ValueError(
            f"Input data must have two dimensions, but found shape: {x.shape}"
        )


def _check_is_np3d(x):
    if not isinstance(x, np.ndarray):
        raise ValueError(
            f"Input data must be an np.ndarray, but found: {type(x)}"
        )
    if x.ndim != 3:
        raise ValueError(
            f"Input data must have three dimensions, but found shape: {x.shape}"
        )


def _check_is_pd2d(x):
    if not isinstance(x, pd.DataFrame):
        raise ValueError(
            f"Input data must be a pd.DataFrame, but found: {type(x)}"
        )
    if x.ndim != 2:
        raise ValueError(
            f"Input data must have two dimensions, but found shape: {x.shape}"
        )


def is_valid_dataframe(x):
    """
    Check if the input is a valid DataFrame.

    Args:
        x: Input data
    Returns:
        bool: Whether the input is a MultiIndex DataFrame
    """
    if not isinstance(x, pd.DataFrame):
        return False
    elif not isinstance(x.columns, pd.MultiIndex):
        return False
    elif x.columns.nlevels != 2:
        return False
    return True


class np2d_to_df(Transformer):
    """
    Convert 2D NumPy array to MultiIndex DataFrame.
    **Don't use this for multivariate time series data**
    """

    def __init__(self, var_name=None, timestamps=None, **kwargs):
        """
        Initializes the converter.

        Args:
            var_name (None or String):
                Optional name to use for the variable
            timestamps (None or list-like with length of n_timepoints):
                Optional time series index of returned DataFrame
        """
        self.var_name = var_name
        self.timestamps = timestamps
        super().__init__(**kwargs)

    def fit(self, x):
        """
        Check if the input data is a 2D NumPy array and create a MultiIndex
        object.

        Args:
            x (ndarray of shape (n_instances, n_timepoints)):
                Input ndarray
        """
        _check_is_np2d(x)

        n_instances, n_timepoints = x.shape
        if self.timestamps is None:
            timestamps = np.arange(n_timepoints)
        else:
            if len(self.timestamps) != n_timepoints:
                raise ValueError(
                    f"Input data has {n_timepoints} timepoints, but only "
                    f"{len(self.timestamps)} steps are supplied in timestamps."
                )

        # create indices for MultiIndex DataFrame
        if self.var_name is None:
            self.mi = pd.MultiIndex.from_product([["var_0"], timestamps])
        else:
            if not isinstance(self.var_name, str):
                raise ValueError(
                    f"var_name must be a String, received type: {type(self.var_name)}"
                )
            self.mi = pd.MultiIndex.from_product([[self.var_name], timestamps])
        super().fit(x)

    def data_transform(self, x):
        """
        Converts input data into a DataFrame with MultiIndex columns
        """
        df = pd.DataFrame(x, columns=self.mi)
        return df


class pd2d_to_df(Transformer):
    """
    Convert 2D DataFrame to a MultiIndex DataFrame.
    """

    def __init__(self, var_name=None, timestamps=None, **kwargs):
        """
        Initializes the converter.

        Args:
            var_name (None or String):
                Optional name to use for the variable
            timestamps (None or list-like with length of n_timepoints):
                Optional time series index of returned DataFrame
        """
        self.var_name = var_name
        self.timestamps = timestamps
        super().__init__(**kwargs)

    def fit(self, x):
        """
        Check if the input data is a pandas DataFrame and create a MultiIndex
        object.

        Args:
            x (DataFrame of shape (n_instances, n_timepoints)):
                Input DataFrame
        """
        _check_is_pd2d(x)
        columns = x.columns
        x = x.to_numpy()

        n_instances, n_timepoints = x.shape
        if self.timestamps is not None:
            if len(self.timestamps) != n_timepoints:
                raise ValueError(
                    f"Input data has {n_timepoints} timepoints, but only "
                    f"{len(self.timestamps)} steps are supplied in timestamps."
                )
            columns = self.timestamps

        # create indices for MultiIndex DataFrame
        if self.var_name is None:
            self.mi = pd.MultiIndex.from_product([["var_0"], columns])
        else:
            if not isinstance(self.var_name, str):
                raise ValueError(
                    f"var_name must be a String, received type: {type(self.var_name)}"
                )
            self.mi = pd.MultiIndex.from_product([[self.var_name], columns])
        super().fit()

    def data_transform(self, x):
        """
        Converts input data into a DataFrame with MultiIndex columns
        """
        df = pd.DataFrame(x, columns=self.mi)
        return df


class np3d_to_df(Transformer):
    """
    Convert 3D NumPy array to a MultiIndex DataFrame.
    """

    def __init__(self, var_names=None, timestamps=None, **kwargs):
        """
        Initializes the converter.

        Args:
            var_names (None or list-like with length of n_variables):
                Optional list of names to use for the variables
            timestamps (None or list-like with length of n_timepoints):
                Optional time series index of returned DataFrame
        """
        self.var_names = var_names
        self.timestamps = timestamps
        super().__init__(**kwargs)

    def fit(self, x):
        """
        Check if the input data is a 3D NumPy array and create a MultiIndex
        object.

        Args:
            x (ndarray with shape (n_instances, n_columns, n_timepoints)):
                Input 3D NumPy array
        """
        _check_is_np3d(x)

        n_instances, n_columns, n_timepoints = x.shape
        if self.var_names is None:
            self.var_names = [f"var_{str(i)}" for i in range(n_columns)]
        else:
            if len(self.var_names) != n_columns:
                raise ValueError(
                    f"Input data has {n_columns} columns, but only "
                    f"{len(self.var_names)} names supplied."
                )
        if self.timestamps is None:
            self.timestamps = np.arange(n_timepoints)
        else:
            if len(self.timestamps) != n_timepoints:
                raise ValueError(
                    f"Input data has {n_timepoints} timepoints, but only "
                    f"{len(self.timestamps)} steps are supplied in timestamps."
                )

        self.mi = pd.MultiIndex.from_product([self.var_names, self.timestamps])
        super().fit(x)

    def data_transform(self, x):
        """
        Converts input data into a DataFrame with MultiIndex columns
        """
        n_instances, n_columns, n_timepoints = x.shape
        flatten_data = x.reshape((n_instances, n_columns*n_timepoints))
        df = pd.DataFrame(flatten_data, columns=self.mi)
        return df


class df_to_np3d(Transformer):
    """
    Convert MultiIndex pandas DataFrame into NumPy ndarray with shape
    (n_instances, n_columns, n_timepoints).
    """

    def data_transform(self, x):
        """
        Convert input DataFrame into 3D NumPy array
        """
        if not is_valid_dataframe(x):
            raise ValueError("Input DataFrame is not a valid DataFrame")
        n_instances = x.index.size
        n_columns, n_timepoints = x.columns.levshape
        array = x.to_numpy().reshape((n_instances, n_columns, n_timepoints))
        return array


class df_to_np2d(df_to_np3d):
    """
    **This should only be used on univariate series**
    Convert MultiIndex pandas DataFrame into NumPy ndarray with shape
    (n_instances, n_timepoints).
    """

    def data_transform(self, x):
        """
        Convert input DataFrame into 2D NumPy array
        """
        array = super().data_transform(x)
        return array.squeeze(axis=1)


# TODO: the following function is basically a wrapper of the above formatters
# We could potentially turn the following into a Transformer if needed in the
# future.
# def check_input(x, univariate=False, to_np=False, to_pd=False):
#     """
#     Validate input data.

#     Args:
#         x (DataFrame or ndarray):
#             Input data
#         univariate (bool):
#             Set this to True if x is univariate.
#         to_np (bool):
#             If True, return x as a 3-dimensional NumPy array.
#         to_pd (bool):
#             If True, return x as a pandas DataFrame with MultiIndex-ed column.

#     Returns:
#         x : pd.DataFrame or np.ndarray
#             Checked and possibly converted input data.
#     """
#     if to_np and to_pd:
#         raise ValueError("`to_np` and `to_pd` cannot both be set to True")
#     # check input type
#     if not isinstance(x, (pd.DataFrame, np.ndarray)):
#         raise ValueError(
#             f"x must be a pd.DataFrame or an np.ndarray, but found: {type(x)}"
#         )
#     if isinstance(x, np.ndarray):
#         if x.ndim == 2:
#             x = x.reshape(x.shape[0], 1, x.shape[1])
#         elif x.ndim == 1 or x.ndim > 3:
#             raise ValueError(
#                 f"If passed as a np.ndarray, x must be a 2 or 3-dimensional "
#                 f"array, but found shape: {x.shape}"
#             )
#         if to_pd:
#             x = numpy3d_to_df(x)
#     # check univariate
#     n_columns = x.shape[1]
#     if univariate and n_columns > 1:
#         raise ValueError(
#             f"x must be univariate with x.shape[1] == 1, but found: "
#             f"x.shape[1] == {n_columns}."
#         )
#     # check pd.DataFrame
#     if isinstance(x, pd.DataFrame):
#         if not is_valid_dataframe(x):
#             raise ValueError(
#                 "If passed as a pd.DataFrame, x must be a MultiIndex pd.DataFrame."
#             )
#         # convert pd.DataFrame
#         if to_np:
#             x = df_to_numpy3d(x)

#     return x
