import pandas as pd

from pyreal.transformers import Transformer


class DataFrameWrapper(Transformer):
    """
    Allows use of standard sklearn transformers while maintaining DataFrame type.
    """

    def __init__(self, wrapped_transformer):
        """
        Initialize the wrapped transformer

        Args:
            wrapped_transformer:
        """
        self.wrapped_transformer = wrapped_transformer

    def fit(self, x):
        """
        Fit the wrapped transformer

        Args:
            x (DataFrame of shape (n_instances, n_features)):
                The dataset to fit to
            **params:
                Additional transformer parameters

        Returns:
            None
        """
        self.wrapped_transformer.fit(x)

    def data_transform(self, x):
        """
        Transform `x` using the wrapped transformer
        Args:
            x (DataFrame of shape (n_instances, n_features)):
                The dataset to transform

        Returns:
            DataFrame of shape (n_instances, n_transformed_features):
                The transformed dataset
        """
        transformed_np = self.wrapped_transformer.transform(x)
        return pd.DataFrame(transformed_np, columns=x.columns, index=x.index)

    def inverse_transform_explanation(self, explanation):
        return explanation

    def transform_explanation(self, explanation):
        """
        For now, always return the explanation, assuming no modifications needed.
        TODO: This will be updated to an AssertionError in GH issue #112.

        Args:
            explanation:
                The explanation to transform

        Returns:
                The unmodified explanation

        """
        return explanation
