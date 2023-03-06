import pandas as pd
import pytest

from pyreal.types.explanations.base import Explanation


class BaseExplanation:
    def __init__(self, a):
        self.a = a


def test_explanation_get():
    base_explanation = BaseExplanation(5)
    explanation = Explanation(base_explanation)

    assert explanation.get() is base_explanation


def test_explanation_get_values():
    base_explanation = BaseExplanation(5)
    base_values = pd.DataFrame([1])
    explanation = Explanation(base_explanation, base_values)

    assert explanation.get() is base_explanation
    assert explanation.get_values() is base_values


def test_explanation_get_values_invalid_type():
    base_explanation = BaseExplanation(5)
    invalid_values = [1]
    with pytest.raises(AssertionError):
        explanation = Explanation(base_explanation, invalid_values)


def test_explanation_get_values_with_no_values_raise_error():
    base_explanation = BaseExplanation(5)
    explanation = Explanation(base_explanation)

    assert explanation.get() is base_explanation
    with pytest.raises(ValueError):
        explanation.get_values()
