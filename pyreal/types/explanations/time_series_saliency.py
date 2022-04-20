import pandas as pd

from pyreal.types.explanations.base import Explanation


class TimeSeriesSaliency(Explanation):
    """
    A type wrapper for time series saliency type outputs from explanation algorithms.
    """

    def validate(self):
        """
        Validate that `self.explanation` is a valid time series saliency map
        Returns:
            None
        Raises:
            AssertionException
                if `self.explanation` is invalid
        """
        super().validate()
        if not isinstance(self.explanation, pd.DataFrame):
            raise AssertionError("DataFrame explanations must be of type DataFrame")
