import pytest
from explanation_types.decision_tree import DecisionTreeExplanation
from sklearn.tree import DecisionTreeClassifier


def test_decision_tree_explanation():
    valid_explanation = DecisionTreeClassifier().fit([[1, 1, 1], [2, 2, 2]], [1, 2])
    explanation = DecisionTreeExplanation(valid_explanation)
    explanation.validate()  # assert does not raise
    assert explanation.get() is valid_explanation

    non_tree_explanation = [1, 1, 1]
    with pytest.raises(AssertionError):
        DecisionTreeExplanation(non_tree_explanation)
    # skip over the init validation to check validate separately
    explanation.explanation = non_tree_explanation
    with pytest.raises(AssertionError):
        explanation.validate()

    unfit_explanation = DecisionTreeClassifier()
    with pytest.raises(AssertionError):
        DecisionTreeExplanation(unfit_explanation)
    # skip over the init validation to check validate separately
    explanation.explanation = non_tree_explanation
    with pytest.raises(AssertionError):
        explanation.validate()
