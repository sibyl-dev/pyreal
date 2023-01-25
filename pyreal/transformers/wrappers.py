import pandas as pd

from pyreal.transformers import Transformer


class DataFrameWrapper(Transformer):
    """
    Allows use of standard sklearn transformers while maintaining DataFrame type.
    """

    def __init__(self, wrapped_transformer, **kwargs):
        """
        Initialize the wrapped transformer

        Args:
            wrapped_transformer:
        """
        self.wrapped_transformer = wrapped_transformer
        super().__init__(**kwargs)

    def fit(self, x, **params):
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
        super().fit(x)

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
