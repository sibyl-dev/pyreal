import reverse_geocoder as rg

from pyreal.transformers import Transformer


class LatLongToPlace(Transformer):
    def __init__(
        self,
        level=0,
        latitude_column="latitude",
        longitude_column="longitude",
        result_column="place",
        **kwargs
    ):
        """
        Initialize the decoder.

        Args:
            level (int):
                Level of place to return, per Geoname's admin levels. 0 will generally be
                neighborhood, 1 will generally be city, and 2 will be state or country.
            latitude_column (string):
                Name of the column containing latitude.
            longitude_column (string):
                Name of the column containing longitude.
            result_column (string):
                Name of column to store the resulting locations in
        """
        if level == 0:
            self.level = "name"
        elif level == 1:
            self.level = "admin2"
        elif level == 2:
            self.level = "admin1"
        else:
            raise ValueError(
                "level must be one of 0 (neighborhood), 1 (city), or 2 (state/country)"
            )
        self.latitude_column = latitude_column
        self.longitude_column = longitude_column
        self.result_column = result_column
        super().__init__(**kwargs)

    def data_transform(self, x):
        if self.latitude_column not in x:
            raise ValueError(
                "Configured latitude column (%s) not present in x" % self.latitude_column
            )
        if self.longitude_column not in x:
            raise ValueError(
                "Configured longitude column (%s) not present in x" % self.longitude_column
            )
        coordinates = list(
            x[[self.latitude_column, self.longitude_column]].itertuples(index=False, name=None)
        )
        results = rg.search(coordinates, verbose=False)
        places = [result[self.level] for result in results]
        x[self.result_column] = places
        return x.drop(columns=[self.latitude_column, self.longitude_column])

    def transform_explanation_feature_based(self, explanation):
        return explanation.combine_columns(
            [self.latitude_column, self.longitude_column], self.result_column
        )
