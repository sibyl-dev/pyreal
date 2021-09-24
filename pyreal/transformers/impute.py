import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from pyreal.transformers import BaseTransformer


class MultiTypeImputer(BaseTransformer):
    """
    Imputes, choosing a strategy based on column type.
    """

    def __init__(self):
        self.numeric_cols = None
        self.categorical_cols = None
        self.numeric_imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
        self.categorical_imputer = SimpleImputer(missing_values=np.nan, strategy="most_frequent")

    def fit(self, x):
        self.numeric_imputer.fit(x[self.numeric_cols])
        self.numeric_cols = x.dropna(axis="columns", how="all") \
            .select_dtypes(include="number").columns
        self.categorical_cols = x.dropna(axis="columns", how="all") \
            .select_dtypes(exclude="number").columns
        if len(self.numeric_cols) == 0 and len(self.categorical_cols) == 0:
            raise ValueError("No valid numeric or categorical cols")
        if len(self.numeric_cols) > 0:
            self.numeric_imputer.fit(x[self.numeric_cols])
        if len(self.categorical_cols) > 0:
            self.categorical_imputer.fit(x[self.categorical_cols])

    def transform(self, x):
        if len(self.categorical_cols) == 0:
            new_numeric_cols = self.numeric_imputer.transform(x[self.numeric_cols])
            return pd.DataFrame(new_numeric_cols, columns=self.numeric_cols, index=x.index)

        if len(self.numeric_cols) == 0:
            new_categorical_cols = self.categorical_imputer.transform(x[self.categorical_cols])
            return pd.DataFrame(new_categorical_cols, columns=self.categorical_cols, index=x.index)

        new_numeric_cols = self.numeric_imputer.transform(x[self.numeric_cols])
        new_categorical_cols = self.categorical_imputer.transform(x[self.categorical_cols])
        return pd.concat([pd.DataFrame(new_numeric_cols, columns=self.numeric_cols, index=x.index),
                          pd.DataFrame(new_categorical_cols, columns=self.categorical_cols,
                                       index=x.index)], axis=1)
