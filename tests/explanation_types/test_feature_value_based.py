import numpy as np
import pytest

from pyreal.explanation_types.feature_value_based import (
    FeatureValueExplanation,
    PartialDependenceExplanation,
)


def test_partial_dependence_explanation_type():
    feature_names = ["f1", "f2"]
    predictions = np.eye(5)[np.newaxis, :, :]
    grid = [np.arange(5), np.arange(5)]

    valid_explanation = FeatureValueExplanation(
        feature_names=feature_names,
        predictions=predictions,
        grid=grid,
    )
    explanation = PartialDependenceExplanation(
        feature_names=feature_names,
        predictions=predictions,
        grid=grid,
    )
    explanation.validate()  # assert does not raise
    assert explanation.get().feature_names is valid_explanation.feature_names
    assert explanation.get().predictions is valid_explanation.predictions
    assert explanation.get().grid is valid_explanation.grid

    non_pdp_explanation = [1, 1, 1]
    # skip over the init validation to check validate separately
    explanation.explanation = non_pdp_explanation
    with pytest.raises((AssertionError, AttributeError)):
        explanation.validate()
