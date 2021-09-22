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
        self.numeric_cols = x.select_dtypes(include="number").columns
        self.categorical_cols = x.select_dtypes(exclude="number").columns
        self.numeric_imputer.fit(x[self.numeric_cols])
        self.categorical_imputer.fit(x[self.categorical_cols])

    def transform(self, x):
        new_numeric_cols = self.numeric_imputer.transform(x[self.numeric_cols])
        new_categorical_cols = self.categorical_imputer.transform(x[self.categorical_cols])
        return pd.concat([pd.DataFrame(new_numeric_cols, columns=self.numeric_cols, index=x.index),
                          pd.DataFrame(new_categorical_cols, columns=self.categorical_cols,
                                       index=x.index)], axis=1)
