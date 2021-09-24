import pandas as pd

from pyreal.transformers import BaseTransformer


class DataFrameWrapper(BaseTransformer):
    """
    Allows use of standard sklearn transformers while maintaining DataFrame type.
    """

    def __init__(self, base_transformer):
        self.base_transformer = base_transformer

    def fit(self, x):
        self.base_transformer.fit(x)

    def transform(self, x):
        transformed_np = self.base_transformer.transform(x)
        return pd.DataFrame(transformed_np, columns=x.columns, index=x.index)
