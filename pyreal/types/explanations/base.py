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
        if values is not None and not isinstance(values, pd.DataFrame) and not isinstance(values, pd.Series):
            raise TypeError("values must be of type DataFrame")
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

    def get_values(self):
        """
        Return the values associated with the explanation

        Returns:
            DataFrame of shape (n_instances, n_features)
                The values associated with this explanation
        """
        if self.values is None:
            raise ValueError("This explanation type does not include values")
        return self.values

    def validate(self):
        """
        Validate that `self.explanation` is a valid object of type `Explanation`
        Returns:
            None
        Raises:
            AssertionException
                if `self.explanation` is invalid
        """
