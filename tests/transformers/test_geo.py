import pandas as pd
from pyreal.transformers import LatLongToPlace
from pandas.testing import assert_frame_equal


def test_latlongtoplace():
    latlongdf = pd.DataFrame(
        [[51.51, -0.11], [38.895, -77.037]], columns=["Latitude", "Longitude"]
    )
    transformer = LatLongToPlace(level=0, latitude_column="Latitude", longitude_column="Longitude")

    result = transformer.transform(latlongdf)
    expected = pd.DataFrame([["City of Westminster"], ["Washington, D.C."]], columns=["place"])
    assert_frame_equal(result, expected)
