from pyreal.transformers import Transformer


class DimensionAdder(Transformer):
    """
    Adds an additional dimension to univariate time series data
        (n_instances, n_timesteps) -> (n_instances, n_timesteps, 1)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def data_transform(self, x):
        return x.to_numpy().reshape(x.shape[0], x.shape[1], 1)
