import pandas as pd

data = pd.read_csv("data_small.csv")
data["Exterior1st"].replace(
    {
        "BrkComm": "Brick",
        "CBlock": "Cinder Block",
        "WdShing": "Wood Shingles",
        "AsbShng": "Asbestos Shingles",
        "Stucco": "Stucco",
        "AsphShn": "Asphalt Shingles",
        "Wd Sdng": "Wood Siding",
        "MetalSd": "Metal Siding",
        "HdBoard": "Hard Board",
        "Plywood": "Plywood",
        "VinylSd": "Vinyl Siding",
        "CemntBd": "Cement Board",
        "Stone": "Stone",
        "ImStucc": "Imitation Stucco",
        "BrkFace": "Brick Face",
    },
    inplace=True,
)
neighborhoods = {
    "Blmngtn": "Bloomington Heights",
    "Blueste": "Bluestem",
    "BrDale": "Briardale",
    "BrkSide": "Brookside",
    "ClearCr": "Clear Creek",
    "CollgCr": "College Creek",
    "Crawfor": "Crawford",
    "Edwards": "Edwards",
    "Gilbert": "Gilbert",
    "IDOTRR": "Iowa DOT and Rail Road",
    "MeadowV": "Meadow Village",
    "Mitchel": "Mitchell",
    "NAmes": "North Ames",
    "NoRidge": "Northridge",
    "NPkVill": "Northpark Villa",
    "NridgHt": "Northridge Heights",
    "NWAmes": "Northwest Ames",
    "OldTown": "Old Town",
    "SWISU": "South & West of Iowa State University",
    "Sawyer": "Sawyer",
    "SawyerW": "Sawyer West",
    "Somerst": "Somerset",
    "StoneBr": "Stone Brook",
    "Timber": "Timberland",
    "Veenker": "Veenker",
}
data["Neighborhood"].replace(neighborhoods, inplace=True)
data.to_csv("data_small.csv", index=False)
