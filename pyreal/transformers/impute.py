import numpy as np
import pandas as pd

from pyreal.transformers import Transformer


class MultiTypeImputer(Transformer):
    """
    Imputes a data set, handling columns of different types. Imputes numeric columns with the mean,
    and categorical columns with the mode value.
    """

    def __init__(self, columns=None, **kwargs):
        """
        Initialize the base imputers
        """
        if columns is not None and not isinstance(columns, (list, tuple, np.ndarray, pd.Index)):
            columns = [columns]
        self.columns = columns

        self.numeric_cols = None
        self.categorical_cols = None
        self.means = None
        self.modes = None
        super().__init__(**kwargs)

    def fit(self, x, **params):
        """
        Fit the imputer

        Args:
            x (DataFrame of shape (n_instances, n_features)):
                The dataset to fit to

        Returns:
            None
        """
        if self.columns is None:
            self.columns = x.columns

        self.numeric_cols = (
            x[self.columns]
            .dropna(axis="columns", how="all")
            .select_dtypes(include="number")
            .columns
        )
        self.categorical_cols = (
            x[self.columns]
            .dropna(axis="columns", how="all")
            .select_dtypes(exclude="number")
            .columns
        )

        self.means = x[self.numeric_cols].mean(axis=0)
        self.modes = x[self.categorical_cols].mode(axis=0)
        if self.modes.shape[0] > 0:
            self.modes = self.modes.iloc[0, :]

        super().fit(x)

    def data_transform(self, x):
        """
        Imputes `x`. Numeric columns get imputed with the column mean. Categorical columns get
        imputed with the column mode.
        Args:
            x (DataFrame of shape (n_instances, n_features)):
                The dataset to impute

        Returns:
            DataFrame of shape (n_instances, n_transformed_features):
                The imputed dataset
        """
        if self.numeric_cols is None:
            raise RuntimeError("Must fit imputer before transforming")
        types = x[self.columns].dtypes
        series_flag = False
        name = None
        if isinstance(x, pd.Series):
            series_flag = True
            name = x.name
            x = x.to_frame().T

        result = x.copy()
        result[self.numeric_cols] = result[self.numeric_cols].fillna(value=self.means)
        result[self.categorical_cols] = result[self.categorical_cols].fillna(value=self.modes)

        result = result.astype(types)
        if series_flag:
            result = result.squeeze()
            result.name = name
        return result
