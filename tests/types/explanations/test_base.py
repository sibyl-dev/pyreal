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
        Explanation(base_explanation, invalid_values)


def test_update_values():
    base_explanation = BaseExplanation(5)
    orig_values = pd.DataFrame([1])
    new_values = pd.DataFrame([2])
    invalid_values = [1]

    explanation = Explanation(base_explanation, orig_values)

    assert explanation.get_values() is orig_values

    explanation.update_values(new_values)
    assert explanation.get_values() is new_values

    with pytest.raises(AssertionError):
        explanation.update_values(invalid_values)


def test_update_explanation():
    orig_explanation = BaseExplanation(5)
    orig_values = pd.DataFrame([1])
    new_explanation = pd.DataFrame([2])

    explanation = Explanation(orig_explanation, orig_values)

    assert explanation.get() is orig_explanation

    explanation.update_explanation(new_explanation)
    assert explanation.get() is new_explanation


def test_get_all():
    base_explanation = BaseExplanation(5)
    orig_values = pd.DataFrame([1])

    explanation1 = Explanation(base_explanation, orig_values)
    explanation2 = Explanation(*explanation1.get_all())

    assert explanation2.get() is base_explanation
    assert explanation2.get_values() is orig_values
