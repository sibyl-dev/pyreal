.. _pyreal.explanation_types:

Explanation Type
=========================
.. currentmodule:: pyreal.types.explanations

Explanation Type
~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: api/

    base.Explanation
    base.Explanation.get
    base.Explanation.validate

DataFrame Explanation Type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: api/

    feature_based.FeatureBased
    feature_based.FeatureBased.get
    feature_based.FeatureBased.validate

Feature Importance Explanation Type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: api/

    feature_based.FeatureImportanceExplanation
    feature_based.FeatureImportanceExplanation.get
    feature_based.FeatureImportanceExplanation.validate

Additive Feature Importance Explanation Type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: api/

    feature_based.AdditiveFeatureImportanceExplanation
    feature_based.AdditiveFeatureImportanceExplanation.get
    feature_based.AdditiveFeatureImportanceExplanation.validate

Feature Contribution Explanation Type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: api/

    feature_based.FeatureContributionExplanation
    feature_based.FeatureContributionExplanation.get
    feature_based.FeatureContributionExplanation.validate

Additive Feature Contribution Explanation Type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: api/

    feature_based.AdditiveFeatureContributionExplanation
    feature_based.AdditiveFeatureContributionExplanation.get
    feature_based.AdditiveFeatureContributionExplanation.validate

Feature Value Based Explanation Type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: api/

    feature_value_based.FeatureValueBased
    feature_value_based.FeatureValueBased.get
    feature_value_based.FeatureValueBased.validate
    feature_value_based.FeatureValueBased.update_feature_names

Partial Dependence Explanation Type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: api/

    feature_value_based.PartialDependenceExplanation
    feature_value_based.PartialDependenceExplanation.get
    feature_value_based.PartialDependenceExplanation.validate
    feature_value_based.PartialDependenceExplanation.update_feature_names

Decision Tree Explanation Type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: api/

    decision_tree.DecisionTreeExplanation
    decision_tree.DecisionTreeExplanation.get
    decision_tree.DecisionTreeExplanation.validate
