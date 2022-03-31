import numpy as np
import pandas as pd

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
