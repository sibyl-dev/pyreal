import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

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
        self.numeric_imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
        self.categorical_imputer = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
        self.types = None
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
            x[self.columns].dropna(axis="columns", how="all").select_dtypes(include="number").columns
        )
        self.categorical_cols = (
            x[self.columns].dropna(axis="columns", how="all").select_dtypes(exclude="number").columns
        )
        if len(self.numeric_cols) == 0 and len(self.categorical_cols) == 0:
            raise ValueError("No valid numeric or categorical cols")
        if len(self.numeric_cols) > 0:
            self.numeric_imputer.fit(x[self.numeric_cols])
        if len(self.categorical_cols) > 0:
            self.categorical_imputer.fit(x[self.categorical_cols])

        self.types = x[self.columns].dtypes.to_dict()
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
        if self.types is None:
            raise RuntimeError("Must fit imputer before transforming")
        series_flag = False
        name = None
        if isinstance(x, pd.Series):
            series_flag = True
            name = x.name
            x = x.to_frame().T
        x[self.columns] = x[self.columns].fillna(value=np.nan)
        if len(self.categorical_cols) == 0:
            new_numeric_cols = self.numeric_imputer.transform(x[self.numeric_cols])
            result = pd.DataFrame(new_numeric_cols, columns=self.numeric_cols, index=x.index)

        elif len(self.numeric_cols) == 0:
            new_categorical_cols = self.categorical_imputer.transform(x[self.categorical_cols])
            result = pd.DataFrame(
                new_categorical_cols, columns=self.categorical_cols, index=x.index
            )

        else:
            new_numeric_cols = self.numeric_imputer.transform(x[self.numeric_cols])
            new_categorical_cols = self.categorical_imputer.transform(x[self.categorical_cols])
            result = pd.concat(
                [
                    pd.DataFrame(new_numeric_cols, columns=self.numeric_cols, index=x.index),
                    pd.DataFrame(
                        new_categorical_cols, columns=self.categorical_cols, index=x.index
                    ),
                ],
                axis=1,
            )
        print(result)
        print(x.dtypes.to_dict())
        result = result.astype(self.types)
        if series_flag:
            result = result.squeeze()
            result.name = name
        return result
