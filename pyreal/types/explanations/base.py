import pandas as pd


class Explanation:
    """
    A type wrapper for outputs from explanation algorithms. Validates that an object is a
    valid explanation output.
    """

    def __init__(self, explanation, values=None):
        """
        Set the wrapped explanation to `explanation` and values to `values` and validate
        Args:
            explanation (object):
                wrapped explanation
            values (DataFrame of shape (n_instances, n_features)):
                Values corresponding with the object being explained

        """
        self.explanation = explanation
        self.values = values
        self.validate()

    def get(self):
        """
        Get the explanation wrapped by this type
        Returns:
            object
                wrapped explanation object
        """
        return self.explanation

    def get_all(self):
        """
        Get the explanation and wrapped values as a tuple. Convenience function for making a new
        Explanation from a previous one.

        Returns:
            tuple
                Explanation and values
        """
        return self.explanation, self.values

    def get_values(self):
        """
        Return the values associated with the explanation

        Returns:
            DataFrame of shape (n_instances, n_features)
                The values associated with this explanation
        """
        return self.values

    def update_values(self, values):
        """
        Updates this objects values, and validates

        Args:
            values (DataFrame of shape (n_instances, n_features)):
                New values
        """
        self.values = values
        self.validate()

    def validate(self):
        """
        Validate that `self.explanation` is a valid object of type `Explanation`
        Returns:
            None
        Raises:
            AssertionException
                if `self.explanation` is invalid
        """
        if (
            self.values is not None
            and not isinstance(self.values, pd.DataFrame)
            and not isinstance(self.values, pd.Series)
        ):
            raise AssertionError("values must be of type DataFrame")
