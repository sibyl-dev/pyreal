import pandas as pd

from pyreal.explanation_types.explanations.base import Explanation, convert_columns_with_dict


class ExampleBasedExplanation(Explanation):
    """
    A type wrapper for example-based type outputs from explanation algorithms.

    Example-based types include dictionary linking input rows to DataFrames, and optionally
     a second dictionary linking the same input rows to Series, where the DataFrames are examples
     (X) for the corresponding row and the Series are the corresponding target (y) values.
     The row order wil depend on the specific input type.
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
        try:
            example_dict = self.explanation[0]
            target_dict = self.explanation[1]
        except TypeError:
            raise AssertionError(
                "Example explanations expect a tuple of example-dictionary and target-dictionary"
            )
        if not isinstance(example_dict, dict):
            raise AssertionError("Example explanations must contain a dictionary of DataFrames")
        for row_id in example_dict:
            if not isinstance(example_dict[row_id], pd.DataFrame):
                raise AssertionError(
                    "All items in example explanations' example dicts must be DataFrames."
                )
        if target_dict is not None and not isinstance(target_dict, dict):
            raise AssertionError("Example explanation given invalid target dict.")
        for row_id in target_dict:
            if not isinstance(target_dict[row_id], pd.Series):
                raise AssertionError(
                    "All items in example explanations' target dicts must be Series."
                )

        super().validate()

    def apply_feature_descriptions(self, feature_descriptions):
        def func(df):
            return convert_columns_with_dict(df, feature_descriptions)

        self.update_examples(func)
        super().apply_feature_descriptions(feature_descriptions)

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
            return self.get()[0][row_id]
        return self.get()[0][row_id].iloc[rank]

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
            return self.get()[1][row_id]
        return self.get()[1][row_id].iloc[rank]

    def get_row_ids(self):
        """
        Return all row_ids held by this explanation
        """
        return self.get()[0].keys()

    def update_examples(self, func):
        for key in self.get()[0]:
            self.get()[0][key] = func(self.get()[0][key])


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
        if self.explanation[1] is None:
            raise AssertionError("Similar example explanations must come with target values.")
        super().validate()


class CounterfactualExplanation(ExampleBasedExplanation):
    """
    A type wrapper for explanations that include most similar rows from the training set.

    Contains a dict of dataframes
    """

    def __init__(self, explanation, values=None):
        """
        Set the wrapped explanation to `explanation` and values to `values` and validate
        Args:
            explanation (object):
                wrapped explanation
            values (DataFrame of shape (n_instances, n_features) or None):
                Values corresponding with the object being explained

        """
        explanation = (explanation, None)
        super().__init__(explanation, values)

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
