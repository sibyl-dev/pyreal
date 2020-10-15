.. _sibyl.explainer

Explainer
================
.. currentmodule:: real.explainers

Base Explainer
~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: api/

    Explainer
    Explainer.fit
    Explainer.produce
    Explainer.transform_to_x_explain
    Explainer.transform_to_x_model
    Explainer.transform_to_x_interpret
    Explainer.model_predict
    Explainer.feature_description

Local Feature Contribution Explainer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: api/

    LocalFeatureContributionsBase
    LocalFeatureContributionsBase.fit
    LocalFeatureContributionsBase.produce
    LocalFeatureContributionsBase.transform_contributions
    LocalFeatureContribution
    LocalFeatureContribution.fit
    LocalFeatureContribution.produce
    ShapFeatureContribution
    ShapFeatureContribution.fit
    ShapFeatureContribution.produce
    ShapFeatureContribution.get_contributions
    ShapFeatureContribution.transform_contributions
    fit_and_produce_local_feature_contributions
    fit_local_feature_contributions
    produce_feature_contributions


