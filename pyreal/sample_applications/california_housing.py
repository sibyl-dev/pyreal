import os
from urllib.parse import urljoin

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "data_cal_housing")
DATA_FILE = os.path.join(DATA_DIR, "california.csv")
CITY_FILE = os.path.join(DATA_DIR, "cal_cities_lat_long.csv")
MODEL_FILE = os.path.join(DATA_DIR, "model.pkl")
TRANSFORMER_FILE = os.path.join(DATA_DIR, "transformers.pkl")
AWS_BASE_URL = "https://pyreal-data.s3.amazonaws.com/"


def load_feature_descriptions():
    return {
        "longitude": "Longitude",
        "latitude": "Latitude",
        "housing_median_age": "Median house age",
        "total_rooms": "Total Number of Rooms",
        "total_bedrooms": "Total Number of Bedrooms",
        "population": "Population",
        "households": "Number of Households",
        "median_income": "Median Income",
        "ocean_proximity": "Proximity to Ocean",
        "neighborhood": "Neighborhood",
    }


def load_data(n_rows=None):
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
    else:
        url = urljoin(AWS_BASE_URL, "usability_study/california.csv")
        print(url)
        df = pd.read_csv(url)

        if not os.path.isdir(DATA_DIR):
            os.mkdir(DATA_DIR)
        df.to_csv(DATA_FILE, index=False)
    y = df["median_house_value"]
    x_orig = df.drop("median_house_value", axis="columns")
    if n_rows is not None:
        return x_orig[:n_rows], y[:n_rows]
    return x_orig, y


def load_city_data():
    if os.path.exists(CITY_FILE):
        df = pd.read_csv(CITY_FILE)
    else:
        url = urljoin(AWS_BASE_URL, "usability_study/cal_cities_lat_long.csv")
        df = pd.read_csv(url)

        if not os.path.isdir(DATA_DIR):
            os.mkdir(DATA_DIR)
        df.to_csv(CITY_FILE, index=False)
    return df
