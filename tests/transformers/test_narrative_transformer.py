import pandas as pd

from pyreal import RealApp
from pyreal.explainers import LocalFeatureContribution
from pyreal.explanation_types import FeatureContributionExplanation
from pyreal.transformers import NarrativeTransformer


def test_transform_feature_contribution(mock_narrator):
    contributions = pd.DataFrame(
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]], columns=["A", "B", "C"]
    )
    values = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=["A", "B", "C"])
    explanation = FeatureContributionExplanation(contributions, values)
    transformer = NarrativeTransformer(narrator=mock_narrator["narrator"])
    transformed_explanation = transformer.transform_explanation(explanation).get()
    assert len(transformed_explanation) == contributions.shape[0]
    for response in transformed_explanation:
        assert response == mock_narrator["response"]


def test_transform_feature_contribution_full_workflow(regression_one_hot, mock_narrator):
    model = regression_one_hot
    x = model["x"]
    lfc = LocalFeatureContribution(
        model=model["model"],
        e_algorithm="shap",
        transformers=[model["transformers"]]
        + [NarrativeTransformer(narrator=mock_narrator["narrator"])],
    )
    lfc.fit(x)

    x_one_dim = pd.DataFrame([[2, 10, 10]], columns=["A", "B", "C"])
    x_multi_dim = pd.DataFrame([[4, 1, 1], [6, 2, 3]], columns=["A", "B", "C"])
    contributions = lfc.produce(x_one_dim).get()
    assert len(contributions) == 1
    assert contributions[0] == mock_narrator["response"]

    contributions = lfc.produce(x_multi_dim).get()
    assert len(contributions) == 2
    assert contributions[0] == mock_narrator["response"]
    assert contributions[1] == mock_narrator["response"]


def test_transform_feature_contribution_realapp(regression_one_hot, mock_narrator):
    real_app = RealApp(
        regression_one_hot["model"],
        regression_one_hot["x"],
        transformers=[regression_one_hot["transformers"]]
        + [NarrativeTransformer(interpret=True, model=False, narrator=mock_narrator["narrator"])],
        id_column="ID",
    )

    features = ["A", "B", "C"]
    x_one_dim = pd.Series([4, 1, 1, "ab"], index=features + ["ID"])
    explanation = real_app.produce_feature_contributions(x_one_dim)
    assert len(explanation) == 1
    assert explanation[0] == mock_narrator["response"]

    x_multi_dim = pd.DataFrame([[4, 1, 1, "a"], [6, 2, 3, "b"]], columns=features + ["ID"])

    explanation = real_app.produce_feature_contributions(x_multi_dim)
    assert len(explanation) == 2
    assert explanation["a"] == mock_narrator["response"]
    assert explanation["b"] == mock_narrator["response"]
