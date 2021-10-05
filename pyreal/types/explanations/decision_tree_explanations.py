from pyreal.types.explanations.base import ExplanationType

from sklearn.tree import BaseDecisionTree


class DecisionTreeExplanationType(ExplanationType):
    @staticmethod
    def validate(explanation):
        if not isinstance(explanation, BaseDecisionTree):
            raise AssertionError("Decision tree explanations must be sklearn Decision Trees")
