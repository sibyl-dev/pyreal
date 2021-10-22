from pyreal.types.explanations.base_explanation import ExplanationType


class BaseExplanation:
    def __init__(self, a):
        self.a = a


def test_explanation_type():
    base_explanation = BaseExplanation(5)
    explanation = ExplanationType(base_explanation)

    assert explanation.get() == base_explanation
