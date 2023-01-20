"""
The default and preferred format for multi-variable time-series data in Pyreal is
a DataFrame with two levels of columns. The first level of columns represents the
variables of the data, and the second level represents the number of timesteps in
the signal(s). The Index of this DataFrame numerates the number of instances in
the data.

Let `data` be a time-series dataset in Pyreal format with 100 instances, 3 variables
and 200 steps. The indexing schemes of Pyreal time-series format work similarly
as Pandas DataFrame.

Examples
-------------------
Access the first instance sequence of data
>>> data.iloc[0]
Access the value of variable "dim_0" at step 50
>>> data["dim_0"][50]
or
>>> data["dim_0", 50]

The indexing on instances and variables can also be used together
>>> data["dim_0", 50].iloc[0]
"""
import numpy as np
import pandas as pd

from pyreal.transformers import Transformer, fit_transformers, run_transformers


def _check_is_np2d(x):
    if not isinstance(x, np.ndarray):
        raise TypeError(f"Input data must be an np.ndarray, but found: {type(x)}")
    if x.ndim != 2:
        raise ValueError(f"Input data must have two dimensions, but found shape: {x.shape}")


def _check_is_np3d(x):
    if not isinstance(x, np.ndarray):
        raise TypeError(f"Input data must be an np.ndarray, but found: {type(x)}")
    if x.ndim != 3:
        raise ValueError(f"Input data must have three dimensions, but found shape: {x.shape}")


def _check_is_pd2d(x):
    if not isinstance(x, pd.DataFrame):
        raise TypeError(f"Input data must be a pd.DataFrame, but found: {type(x)}")
    if x.ndim != 2:
        raise ValueError(f"Input data must have two dimensions, but found shape: {x.shape}")


def _check_is_pyreal_df(x):
    if not is_valid_dataframe(x):
        raise TypeError(
            f"Input data must be a MultiIndex-ed column DataFrame, but found: {type(x)}"
        )


def _check_is_sktime_nest(x):
    if not isinstance(x, pd.DataFrame):
        raise TypeError(f"Input data must be a pd.DataFrame, but found: {type(x)}")
    else:
        if not x.applymap(lambda cell: isinstance(cell, pd.Series)).values.any():
            raise ValueError("Entries of input data must be pd.Series")


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


class Numpy2dToMultiIndexFrame(Transformer):
    """
    **This Transformer should only be used on univariate time series data**
    Convert 2D NumPy array to Pyreal time-series format.
    """

    def __init__(self, var_name=None, timestamps=None, **kwargs):
        """
        Initializes the converter.

        Args:
            var_name (None or String):
                Optional name to use for the variable
            timestamps (None or list-like with length of n_timepoints):
                Optional time series index for returned MultiIndex DataFrame
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
            self.timestamps = np.arange(n_timepoints)
        else:
            if len(self.timestamps) != n_timepoints:
                raise ValueError(
                    f"Input data has {n_timepoints} timepoints, but only "
                    f"{len(self.timestamps)} steps are supplied in timestamps."
                )

        # create indices for MultiIndex DataFrame
        if self.var_name is None:
            self.mi = pd.MultiIndex.from_product([["var_0"], self.timestamps])
        else:
            if isinstance(self.var_name, str):
                self.mi = pd.MultiIndex.from_product([[self.var_name], self.timestamps])
            elif isinstance(self.var_name, (list, pd.Index)):
                self.mi = pd.MultiIndex.from_product([self.var_name, self.timestamps])
            else:
                raise TypeError(f"var_name must be a String, received type: {type(self.var_name)}")
        super().fit(x)

    def data_transform(self, x):
        """
        Converts input data into a DataFrame with MultiIndex columns

        Args:
            x (ndarray of shape (n_instances, n_timepoints)):
                Input ndarray
        """
        _check_is_np2d(x)
        df = pd.DataFrame(x, columns=self.mi)
        return df


class Pandas2dToMultiIndexFrame(Transformer):
    """
    Convert 2D DataFrame to Pyreal time-series format.
    """

    def __init__(self, var_name=None, timestamps=None, **kwargs):
        """
        Initializes the converter.

        Args:
            var_name (None or String):
                Optional name to use for the variable
            timestamps (None or list-like with length of n_timepoints):
                Optional time series index for returned DataFrame
        """
        self.var_name = var_name
        self.timestamps = timestamps
        super().__init__(**kwargs)

    def fit(self, x):
        """
        Check if the input data is a Pandas DataFrame and create a MultiIndex
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
                raise TypeError(f"var_name must be a String, received type: {type(self.var_name)}")
            self.mi = pd.MultiIndex.from_product([[self.var_name], columns])
        super().fit()

    def data_transform(self, x):
        """
        Converts input data into a DataFrame with MultiIndex columns

        Args:
            x (DataFrame of shape (n_instances, n_timepoints)):
                Input DataFrame
        """
        _check_is_pd2d(x)
        df = pd.DataFrame(x, columns=self.mi)
        return df


class Numpy3dToMultiIndexFrame(Transformer):
    """
    Convert 3D NumPy array to Pyreal time-series format.
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
                raise TypeError(
                    f"Input data has {n_timepoints} timepoints, but only "
                    f"{len(self.timestamps)} steps are supplied in timestamps."
                )

        self.mi = pd.MultiIndex.from_product([self.var_names, self.timestamps])
        super().fit(x)

    def data_transform(self, x):
        """
        Converts input data into a DataFrame with MultiIndex column

        Args:
            x (ndarray with shape (n_instances, n_columns, n_timepoints)):
                Input 3D NumPy array
        """
        _check_is_np3d(x)
        n_instances, n_columns, n_timepoints = x.shape
        flatten_data = x.reshape((n_instances, n_columns * n_timepoints))
        df = pd.DataFrame(flatten_data, columns=self.mi)
        return df


class MultiIndexFrameToNumpy3d(Transformer):
    """
    Convert Pyreal time-series format into NumPy ndarray with shape
    (n_instances, n_columns, n_timepoints).
    """

    def data_transform(self, x):
        """
        Convert input DataFrame into 3D NumPy array
        """
        _check_is_pyreal_df(x)
        n_instances = x.index.size
        n_columns, n_timepoints = x.columns.levshape
        array = x.to_numpy().reshape((n_instances, n_columns, n_timepoints))
        return array


class MultiIndexFrameToNumpy2d(MultiIndexFrameToNumpy3d):
    """
    **This Transformer should only be used on univariate series**
    Convert Pyreal time-series format into NumPy ndarray with shape
    (n_instances, n_timepoints).
    """

    def data_transform(self, x):
        """
        Convert input DataFrame into 2D NumPy array
        """
        array = super().data_transform(x)
        return array.squeeze(axis=1)


class NestedFrameToMultiIndexFrame(Transformer):
    """
    Convert sktime nested DataFrame format into Pyreal time-series format.
    """

    def fit(self, x):
        """
        Check if the input data is a sktime nested DataFrame and create a
        MultiIndex object.

        Args:
            x (pandas DataFrame):
                Input sktime nested DataFrame
        """
        _check_is_sktime_nest(x)
        # use the values of an instance of a single variable as inference
        sample_series = x.iloc[0][0]

        self.var_names = x.columns
        self.timestamps = sample_series.index

        self.mi = pd.MultiIndex.from_product([self.var_names, self.timestamps])
        super().fit(x)

    def data_transform(self, x):
        """
        Converts input data into a DataFrame with MultiIndex columns

        Args:
            x (pandas DataFrame):
                Input sktime nested DataFrame

        Returns:
            MultiIndex DataFrame
        """
        # The following mask is for handling missing values in sktime
        # reserved for reference.
        # nested_col_mask = [*x.applymap(
        #     lambda cell: isinstance(cell, (pd.Series, np.ndarray))
        #     ).values]
        _check_is_sktime_nest(x)
        n_instances, n_columns = x.shape
        n_timepoints = x.iloc[0][0].size
        data = []
        for i in x.index:
            # single variable data at time i
            single_data_ti = x.loc[i, :].to_numpy()
            multi_data_ti = [np.array(s) for s in single_data_ti]
            data_ti = np.concatenate(multi_data_ti, axis=0)

            data.append(data_ti)

        full_data = np.concatenate(data, axis=0).reshape((n_instances, n_columns * n_timepoints))
        df = pd.DataFrame(full_data, columns=self.mi)

        return df


class NestedFrameToNumpy3d(Transformer):
    """
    Convert sktime nested DataFrame format into NumPy ndarray
    with shape (n_instances, n_variables, n_timepoints).
    """

    def data_transform(self, x):
        """
        Args:
            x (pandas DataFrame):
                Input sktime nested DataFrame

        Returns:
            NumPy ndarray, converted NumPy ndarray
        """
        _check_is_sktime_nest(x)
        return np.stack(
            x.applymap(lambda cell: cell.to_numpy())
            .apply(lambda row: np.stack(row), axis=1)
            .to_numpy()
        )


class MultiIndexFrameToNestedFrame(Transformer):
    """
    Convert Pyreal time-series format to sktime nested DataFrame.
    """

    def data_transform(self, x):
        """
        Convert input DataFrame into sktime nested DataFrame.
        """
        _check_is_pyreal_df(x)
        np_data = x.to_numpy()

        columns = x.columns.get_level_values(0).unique()
        timestamps = x.columns.get_level_values(1).unique()
        instance_idxs = x.index
        x_nested = pd.DataFrame(columns=columns, index=instance_idxs)

        x_3d = np_data.reshape((x.shape[0], columns.size, timestamps.size))
        for vidx, var in enumerate(columns):
            x_nested[var] = [
                pd.Series(x_3d[i, vidx, :], index=timestamps) for i in range(len(instance_idxs))
            ]

        return x_nested


class Numpy2dToNestedFrame(Transformer):
    """
    Convert 2D NumPy array to sktime nested DataFrame.
    """

    def __init__(self, var_name=None, timestamps=None, **kwargs):
        """
        Initializes the converter.

        Args:
            var_name (None or String):
                Optional name to use for the variable
            timestamps (None or list-like with length of n_timepoints):
                Optional time series index of returned nested DataFrame
        """
        self._transformers = [
            Numpy2dToMultiIndexFrame(var_name=var_name, timestamps=timestamps, **kwargs),
            MultiIndexFrameToNestedFrame(),
        ]
        super().__init__(**kwargs)

    def fit(self, x):
        """
        Check if the input data is a NumPy 2D array

        Args:
            x (ndarray of shape (n_instances, n_timepoints)):
                Input ndarray
        """
        fit_transformers(self._transformers, x)

    def data_transform(self, x):
        """
        Convert input NumPy 2D array into sktime nested DataFrame.
        """
        return run_transformers(self._transformers, x)


class Numpy3dToNestedFrame(Transformer):
    """
    Convert 3D NumPy array to sktime nested DataFrame.
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
        self._transformers = [
            Numpy3dToMultiIndexFrame(var_names=var_names, timestamps=timestamps, **kwargs),
            MultiIndexFrameToNestedFrame(),
        ]
        super().__init__(**kwargs)

    def fit(self, x):
        """
        Check if the input data is a NumPy 3D array

        Args:
            x (ndarray of shape (n_instances, n_timepoints)):
                Input ndarray
        """
        fit_transformers(self._transformers, x)

    def data_transform(self, x):
        """
        Convert input NumPy 3D array into sktime nested DataFrame.
        """
        return run_transformers(self._transformers, x)


class DimensionAdder(Transformer):
    """
    Adds an additional dimension to univariate time series data
        (n_instances, n_timesteps) -> (n_instances, n_timesteps, 1)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def data_transform(self, x):
        return x.to_numpy().reshape(x.shape[0], x.shape[1], 1)
