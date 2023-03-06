from sklearn.exceptions import NotFittedError
from sklearn.tree import BaseDecisionTree
from sklearn.utils.validation import check_is_fitted

from pyreal.types.explanations.base import Explanation


class DecisionTreeExplanation(Explanation):
    """
    A type wrapper for decision-tree based type outputs from explanation algorithms.
    """

    def validate(self):
        """
        Validate that `self.explanation` is a valid and fitted sklearn `DecisionTree`
        Returns:
            None
        Raises:
            AssertionException
                if `self.explanation` is invalid
        """
        if not isinstance(self.explanation, BaseDecisionTree):
            raise AssertionError("Decision tree explanations must be sklearn Decision Trees")
        try:
            check_is_fitted(self.explanation)
        except NotFittedError:
            raise AssertionError(
                "Decision tree explanations must be fitted sklearn Decision Trees"
            )
