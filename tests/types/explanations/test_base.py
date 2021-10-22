from pyreal.types.explanations.base import ExplanationType


class BaseExplanation:
    def __init__(self, a):
        self.a = a


def test_explanation_type():
    base_explanation = BaseExplanation(5)
    explanation = ExplanationType(base_explanation)

    assert explanation.get() is base_explanation
