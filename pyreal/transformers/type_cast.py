from pyreal.transformers import Transformer


class BoolToIntCaster(Transformer):
    def data_transform(self, x):
        """
        Transform booleans to integers
        Args:
            x (DataFrame of shape (n_instances, n_features)):
                The dataset to transform

        Returns:
            DataFrame of shape (n_instances, n_transformed_features):
                The transformed dataset
        """
        return x.replace({False: 0, True: 1})
