import pandas as pd

from pyreal.types.explanations.base import Explanation


class ExampleBased(Explanation):
    """
    A type wrapper for example-based type outputs from explanation algorithms.

    Contains a dict of dataframes
    """

    def validate(self):
        """
        Validate that `self.explanation` is a valid dict of `DataFrames`
        Returns:
            None
        Raises:
            AssertionException
                if `self.explanation` is invalid
        """
        super().validate()


class SimilarExamples(Explanation):
    """
    A type wrapper for explanations that include most similar rows from the training set.

    Contains a dict of dataframes
    """

    def validate(self):
        """
        Validate that `self.explanation` is a valid dict of `DataFrames`
        Returns:
            None
        Raises:
            AssertionException
                if `self.explanation` is invalid
        """
        super().validate()
