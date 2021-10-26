import pandas as pd

from pyreal.transformers import BaseTransformer


class DataFrameWrapper(BaseTransformer):
    """
    Allows use of standard sklearn transformers while maintaining DataFrame type.
    """

    def __init__(self, base_transformer, columns=None):
        self.base_transformer = base_transformer
        self.columns = columns

    def fit(self, x):
        if self.columns is None:
            self.columns = x.columns
        self.base_transformer.fit(x[self.columns])

    def transform(self, x):
        transformed_np = self.base_transformer.transform(x[self.columns])
        transformed = pd.DataFrame(transformed_np, columns=self.columns, index=x.index)
        return pd.concat([x.drop(self.columns, axis="columns"), transformed], axis=1)
