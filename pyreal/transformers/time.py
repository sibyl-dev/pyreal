from pyreal.transformers import Transformer


class TimeTransformer(Transformer):
    """
    A transformer that helps preprocess time-series data 
    """
    def __init__(self, time_column, **kwargs):
        """
        Initializes the transformer

        Args:
            columns (dataframe column label type or list of dataframe column label type):
                Label of column to select, or an ordered list of column labels to select
        """
        
        self.column = time_column
        super().__init__(**kwargs)
    
    def fit(self, x):
        """
        Saves the columns being dropped

        Args:
            x (DataFrame of shape (n_instances, n_features)):
                The dataset to fit on
        Returns:
            None

        """
        self.dropped_columns = list(set(x.columns) - set(self.columns))
        return self

    def data_transform(self, x):
        """
        Reorders and selects the features in x

        Args:
            x (DataFrame of shape (n_instances, n_features)):
                The data to transform
        Returns:
            DataFrame of shape (n_instances, len(columns)):
                The data with features selected and reordered
        """
        return x[self.columns]
