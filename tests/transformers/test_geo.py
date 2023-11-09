import pandas as pd
from pyreal.transformers import LatLongToPlace
from pandas.testing import assert_frame_equal
from pyreal.explanation_types import AdditiveFeatureContributionExplanation


def test_latlongtoplace_data_transform():
    latlongdf = pd.DataFrame(
        [[51.51, -0.11], [38.895, -77.037]], columns=["Latitude", "Longitude"]
    )
    transformer = LatLongToPlace(level=0, latitude_column="Latitude", longitude_column="Longitude")

    result = transformer.transform(latlongdf)
    expected = pd.DataFrame([["City of Westminster"], ["Washington, D.C."]], columns=["place"])
    assert_frame_equal(result, expected)


def test_latlongtoplace_exp_transform():
    latlong_exp = AdditiveFeatureContributionExplanation(
        pd.DataFrame([[10, 15], [3, 5]], columns=["Latitude", "Longitude"])
    )
    transformer = LatLongToPlace(
        level=0, latitude_column="Latitude", longitude_column="Longitude", result_column="City"
    )
    result = transformer.transform_explanation_additive_feature_contribution(latlong_exp)
    expected = AdditiveFeatureContributionExplanation(pd.DataFrame([[25], [8]], columns=["City"]))
    assert_frame_equal(result.get(), expected.get())
