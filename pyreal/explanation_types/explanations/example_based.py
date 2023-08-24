from pyreal.explanation_types.explanations.base import Explanation


class ExampleBasedExplanation(Explanation):
    """
    A type wrapper for example-based type outputs from explanation algorithms.

    Example-based types include dictionary linking input rows to a tuple of (DataFrame, Series),
    where the DataFrame is the set of examples for the corresponding row and the Series is the
    corresponding y values. The DataFrame/Series row order wil depend on the specific input type.
    """

    def validate(self):
        """
        Validate that `self.explanation` is of the expected format.
        Returns:
            None
        Raises:
            AssertionException
                if `self.explanation` is invalid
        """
        super().validate()

    def get_explanation_for_row(self, row_id):
        """
        Get the example explanation generated for the given row_id
        Args:
            row_id:

        Returns:
            A tuple of (DataFrame, Series)
                The examples and corresponding targets for this example
        """
        return self.get()[row_id]

    def get_examples(self, row_id=0, rank=None):
        """
        Get the example in rank-th position for the given row_id.
        Args:
            row_id (int): ID of row to get explanation of.
            rank (int): Which example to return (ie, rank=0 returns the first example generated).
                        If none, return all examples

        Returns:
            DataFrame
                Examples for the chosen row_id
        """
        if rank is None:
            return self.get()[row_id][0]
        return self.get()[row_id][0].iloc[rank]

    def get_targets(self, row_id=0, rank=None):
        """
        Get the targets in rank-th position for the given row_id.
        Args:
            row_id (int): ID of row to get explanation of.
            rank (int): Which example to return (ie, rank=0 returns the first example generated).
                        If none, return all examples

        Returns:
            Series
                targets for the chosen row_id
        """
        if rank is None:
            return self.get()[row_id][1]
        return self.get()[row_id][1].iloc[rank]

    def get_row_ids(self):
        """
        Return all row_ids held by this explanation
        """
        return self.get().keys()

    def update_examples(self, func):
        for key in self.get():
            self.get()[key] = (func(self.get()[key][0]), self.get()[key][1])


class SimilarExampleExplanation(ExampleBasedExplanation):
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
