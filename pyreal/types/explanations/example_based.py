from pyreal.types.explanations.base import Explanation


class ExampleBasedExplanation(Explanation):
    """
    A type wrapper for example-based type outputs from explanation algorithms.
    Example-based types include dictionary linking some key to a tuple of (series, scalar), where
    the series is a row from X and the value is the corresponding y value.

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

    def get_example(self, key, include_target=False):
        if include_target:
            return self.get()[key][0], self.get()[key][1]
        return self.get()[key][0]

    def get_target(self, key):
        return self.get()[key][1]

    def get_all_examples(self, include_targets=False):
        if include_targets:
            return [(self.get()[key][0], self.get()[key][1]) for key in self.get()]
        return [self.get()[key][0] for key in self.get()]

    def get_keys(self):
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
