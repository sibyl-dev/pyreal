from pyreal.transformers import NarrativeTransformer
from pyreal.explanation_types import FeatureContributionExplanation
import pandas as pd
import yaml


def test_transform_feature_contribution():
    with open("../keys.yml", "r") as file:
        config = yaml.safe_load(file)
        openai_api_key = config["openai_api_key"]

    contributions = pd.DataFrame(
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]], columns=["A", "B", "C"]
    )
    values = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=["A", "B", "C"])
    explanation = FeatureContributionExplanation(contributions, values)
    transformer = NarrativeTransformer(openai_api_key=openai_api_key)
    transformed_explanation = transformer.transform_explanation(explanation)
    print(transformed_explanation.get())
