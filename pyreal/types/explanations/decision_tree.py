from sklearn.tree import BaseDecisionTree
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

from pyreal.types.explanations.base import ExplanationType


class DecisionTreeExplanationType(ExplanationType):
    def validate(self):
        if not isinstance(self.explanation, BaseDecisionTree):
            raise AssertionError("Decision tree explanations must be sklearn Decision Trees")
        try:
            check_is_fitted(self.explanation)
        except NotFittedError:
            raise AssertionError(
                "Decision tree explanations must be fitted sklearn Decision Trees")
