import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from pyreal.transformers import Transformer


class MultiTypeImputer(Transformer):
    """
    Imputes a data set, handling columns of different types. Imputes numeric columns with the mean,
    and categorical columns with the mode value.
    """

    def __init__(self, **kwargs):
        """
        Initialize the base imputers
        """
        self.numeric_cols = None
        self.categorical_cols = None
        self.numeric_imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
        self.categorical_imputer = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
        super().__init__(**kwargs)

    def fit(self, x):
        """
        Fit the imputer

        Args:
            x (DataFrame of shape (n_instances, n_features)):
                The dataset to fit to

        Returns:
            None
        """
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
        return self

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

    def transform_explanation(self, explanation):
        """
        Transforms additive contribution explanations. No transformation required.

        Args:
            explanation (ExplanationType):
                The explanation to be transformed

        Returns:
            ExplanationType:
                The transformed explanation
        """
        return explanation

    def inverse_transform_explanation(self, explanation):
        """
        Transforms additive contribution explanations. No transformation required.

        Args:
            explanation (ExplanationType):
                The explanation to be transformed

        Returns:
            ExplanationType:
                The transformed explanation
        """
        return explanation
