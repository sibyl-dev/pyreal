class Explanation:
    """
    A type wrapper for outputs from explanation algorithms. Validates that an object is a
    valid explanation output.
    """

    def __init__(self, *args):
        """
        Set the wrapped explanation to `explanation` and validate
        Args:
            args:
                explanation objects
        """
        self.explanation = args
        self.validate()

    def get(self, ind=0):
        """
        Get the explanation wrapped by this type
        Args:
            ind (int)
                Integer index to get from explanation. If None, return full tuple
        Returns:
            object
                wrapped explanation object
        """
        if ind is None:
            return self.explanation
        return self.explanation[ind]

    def validate(self):
        """
        Validate that `self.explanation` is a valid object of type `Explanation`
        Returns:
            None
        Raises:
            AssertionException
                if `self.explanation` is invalid
        """
