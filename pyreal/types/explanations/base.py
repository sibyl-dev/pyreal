class Explanation:
    """
    A type wrapper for outputs from explanation algorithms. Validates that an object is a
    valid explanation output.
    """

    def __init__(self, explanation):
        """
        Set the wrapped explanation to `explanation` and validate
        Args:
            explanation:
                an explanation algorithm output
        """
        self.explanation = explanation
        self.validate()

    def get(self):
        """
        Get the explanation wrapped by this type

        Returns:
            object
                wrapped explanation object
        """
        return self.explanation

    def validate(self):
        """
        Validate that `self.explanation` is a valid object of type `Explanation`
        Returns:
            None
        Raises:
            AssertionException
                if `self.explanation` is invalid
        """
