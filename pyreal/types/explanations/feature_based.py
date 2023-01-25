import pandas as pd

from pyreal.types.explanations.base import Explanation


class FeatureBased(Explanation):
    """
    A type wrapper for feature-based DataFrame type outputs from explanation algorithms.
    """

    def validate(self):
        """
        Validate that `self.explanation` is a valid `DataFrame`
        Returns:
            None
        Raises:
            AssertionException
                if `self.explanation` is invalid
        """
        super().validate()
        if not isinstance(self.explanation, pd.DataFrame):
            raise AssertionError("DataFrame explanations must be of type DataFrame")


class FeatureImportanceExplanation(FeatureBased):
    """
    A type wrapper for global feature importance FeatureBased type outputs from explanation
    algorithms. Global feature importance explanations give one numeric value per feature,
    representing that feature's overall importance to the model's prediction.
    """

    def validate(self):
        """
        Validate that `self.explanation` is a valid single-row `DataFrame`
        Returns:
            None
        Raises:
            AssertionException
                if `self.explanation` is invalid
        """
        super().validate()
        if self.explanation.shape[0] > 1:
            raise AssertionError("Global Feature Importance Explanations can have only one row")


class AdditiveFeatureImportanceExplanation(FeatureImportanceExplanation):
    """
    A type wrapper for additive global feature importance DataFrame type outputs from explanation
    algorithms. Additive global feature importance give one numeric value per feature,
    representing that feature's overall importance to the model's prediction. Importance values
    can be added together with meaningful effect (ie.,  `importance of feature A + importance of
    feature B = combined importance of feature A and B`)
    """

    def validate(self):
        """
        Validate that `self.explanation` is a valid single-row `DataFrame`
        Returns:
            None
        Raises:
            AssertionException
                if `self.explanation` is invalid
        """
        super().validate()


class FeatureContributionExplanation(FeatureBased):
    """
    A type wrapper for local feature contribution DataFrame type outputs from explanation
    algorithms. Local feature contribution explanations give one numeric value per instance
    per feature, representing that feature's contribution to the model's prediction
    for this instance.
    """

    def validate(self):
        """
        Validate that `self.explanation` is a valid `DataFrame`
        Returns:
            None
        Raises:
            AssertionException
                if `self.explanation` is invalid
        """
        super().validate()


class AdditiveFeatureContributionExplanation(FeatureContributionExplanation):
    """
    A type wrapper for local feature contribution DataFrame type outputs from explanation
    algorithms. Local feature contribution explanations give one numeric value per instance
    per feature, representing that feature's contribution to the model's prediction
    for this instance. Contribution values can be added together with meaningful effect
    (ie.,  `contribution of feature A + contribution of feature B =
    combined contribution of feature A and B`)
    """

    def validate(self):
        """
        Validate that `self.explanation` is a valid `DataFrame`
        Returns:
            None
        Raises:
            AssertionException
                if `self.explanation` is invalid
        """
        super().validate()
