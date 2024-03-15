import pandas as pd

from pyreal.transformers import TransformerBase


class Transformer(TransformerBase):
    """
    Wrap any transformer with a .fit() and .transform() method for use with Pyreal.
    Will convert outputs to DataFrames as needed.
    """

    def __init__(self, wrapped_transformer, columns=None, **kwargs):
        self.wrapped_transformer = wrapped_transformer
        self.columns = columns
        super().__init__(**kwargs)

    def fit(self, x, **params):
        if hasattr(self.wrapped_transformer, "fit"):
            if self.columns is None:
                self.wrapped_transformer.fit(x, **params)
            else:
                self.wrapped_transformer.fit(x[self.columns], **params)
        return super().fit(x)

    def data_transform(self, x):
        if self.columns is None:
            result = self.wrapped_transformer.transform(x)
            if not isinstance(result, pd.DataFrame):
                return pd.DataFrame(result, index=x.index, columns=x.columns)
            return result
        else:
            x = x.copy()
            x[self.columns] = self.wrapped_transformer.transform(x[self.columns])
            return x

    def inverse_data_transform(self, x_new):
        if self.columns is None:
            result = self.wrapped_transformer.inverse_transform(x_new)
            if not isinstance(result, pd.DataFrame):
                return pd.DataFrame(result, index=x_new.index, columns=x_new.columns)
            return result
        else:
            x_new = x_new.copy()
            x_new[self.columns] = self.wrapped_transformer.inverse_transform(x_new[self.columns])
            return x_new

    @staticmethod
    def from_transform_function(transform_func):
        class WrappedTransformer(TransformerBase):
            def data_transform(self, x):
                return transform_func(x)

        return Transformer(WrappedTransformer())
