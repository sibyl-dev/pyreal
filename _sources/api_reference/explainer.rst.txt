.. _pypyreal.explainer

Explainer
================
.. currentmodule:: pyreal.explainers

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
    Explainer.convert_data_to_interpretable

Local Feature Contribution Base
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: api/

    LocalFeatureContributionsBase
    LocalFeatureContributionsBase.fit
    LocalFeatureContributionsBase.produce
    LocalFeatureContributionsBase.get_contributions
    LocalFeatureContributionsBase.transform_contributions

Local Feature Contribution Explainer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: api/

    LocalFeatureContribution
    LocalFeatureContribution.fit
    LocalFeatureContribution.produce

SHAP Feature Contribution Explainer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ShapFeatureContribution
    ShapFeatureContribution.fit
    ShapFeatureContribution.produce
    ShapFeatureContribution.get_contributions
    ShapFeatureContribution.transform_contributions


