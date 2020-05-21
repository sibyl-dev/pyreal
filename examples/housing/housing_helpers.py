# Some code taken from:
#   https://www.kaggle.com/juliencs/a-study-on-regression-applied-to-the-ames-dataset

import pandas as pd
import keras.models
import pickle
import numpy as np
from sklearn.externals import joblib

directory = "../../../Interpretability_User_Studies/housing_user_study/"
model_filename = directory+"model_ridge"
data_filename = directory+"../data/AmesHousing/train.csv"
standardizer_filename = directory+"model_ridge_stdsc"

int_to_cat = {"Alley" : {"Grvl" : 1, "Pave" : 2},
               "BsmtCond" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
               "BsmtExposure" : {"No" : 0, "Mn" : 1, "Av": 2, "Gd" : 3},
               "BsmtFinType1" : {"No" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4,
                                 "ALQ" : 5, "GLQ" : 6},
               "BsmtFinType2" : {"No" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4,
                                 "ALQ" : 5, "GLQ" : 6},
               "BsmtQual" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA": 3, "Gd" : 4, "Ex" : 5},
               "ExterCond" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},
               "ExterQual" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},
               "FireplaceQu" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
               "Functional" : {"Sal" : 1, "Sev" : 2, "Maj2" : 3, "Maj1" : 4, "Mod": 5,
                               "Min2" : 6, "Min1" : 7, "Typ" : 8},
               "GarageCond" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
               "GarageQual" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
               "HeatingQC" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
               "KitchenQual" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
               "LandSlope" : {"Sev" : 1, "Mod" : 2, "Gtl" : 3},
               "LotShape" : {"IR3" : 1, "IR2" : 2, "IR1" : 3, "Reg" : 4},
               "PavedDrive" : {"N" : 0, "P" : 1, "Y" : 2},
               "PoolQC" : {"No" : 0, "Fa" : 1, "TA" : 2, "Gd" : 3, "Ex" : 4},
               "Street" : {"Grvl" : 1, "Pave" : 2},
               "Utilities" : {"ELO" : 1, "NoSeWa" : 2, "NoSewr" : 3, "AllPub" : 4}}


def load_model():
    """
    Load the model
    :return: the model
    """
    try:
        model = keras.models.load_model(model_filename)
        print("loaded keras model...")
    except:
        model = pickle.load(open(model_filename, 'rb'))
        print("loaded pickle model...")
    return model


def load_data():
    data = pd.read_csv(data_filename)
    data = data.drop("Id", axis='columns')
    print("train : " + str(data.shape))
    data = data[data.GrLivArea < 4000]
    y = data.SalePrice

    data.loc[:, "Alley"] = data.loc[:, "Alley"].fillna("None")
    data.loc[:, "BedroomAbvGr"] = data.loc[:, "BedroomAbvGr"].fillna(0)
    data.loc[:, "BsmtQual"] = data.loc[:, "BsmtQual"].fillna("No")
    data.loc[:, "BsmtCond"] = data.loc[:, "BsmtCond"].fillna("No")
    data.loc[:, "BsmtExposure"] = data.loc[:, "BsmtExposure"].fillna("No")
    data.loc[:, "BsmtFinType1"] = data.loc[:, "BsmtFinType1"].fillna("No")
    data.loc[:, "BsmtFinType2"] = data.loc[:, "BsmtFinType2"].fillna("No")
    data.loc[:, "BsmtFullBath"] = data.loc[:, "BsmtFullBath"].fillna(0)
    data.loc[:, "BsmtHalfBath"] = data.loc[:, "BsmtHalfBath"].fillna(0)
    data.loc[:, "BsmtUnfSF"] = data.loc[:, "BsmtUnfSF"].fillna(0)
    data.loc[:, "CentralAir"] = data.loc[:, "CentralAir"].fillna("N")
    data.loc[:, "Condition1"] = data.loc[:, "Condition1"].fillna("Norm")
    data.loc[:, "Condition2"] = data.loc[:, "Condition2"].fillna("Norm")
    data.loc[:, "EnclosedPorch"] = data.loc[:, "EnclosedPorch"].fillna(0)
    data.loc[:, "ExterCond"] = data.loc[:, "ExterCond"].fillna("TA")
    data.loc[:, "ExterQual"] = data.loc[:, "ExterQual"].fillna("TA")
    data.loc[:, "Fence"] = data.loc[:, "Fence"].fillna("No")
    data.loc[:, "FireplaceQu"] = data.loc[:, "FireplaceQu"].fillna("No")
    data.loc[:, "Fireplaces"] = data.loc[:, "Fireplaces"].fillna(0)
    data.loc[:, "Functional"] = data.loc[:, "Functional"].fillna("Typ")
    data.loc[:, "GarageType"] = data.loc[:, "GarageType"].fillna("No")
    data.loc[:, "GarageFinish"] = data.loc[:, "GarageFinish"].fillna("No")
    data.loc[:, "GarageQual"] = data.loc[:, "GarageQual"].fillna("No")
    data.loc[:, "GarageCond"] = data.loc[:, "GarageCond"].fillna("No")
    data.loc[:, "GarageArea"] = data.loc[:, "GarageArea"].fillna(0)
    data.loc[:, "GarageCars"] = data.loc[:, "GarageCars"].fillna(0)
    data.loc[:, "HalfBath"] = data.loc[:, "HalfBath"].fillna(0)
    data.loc[:, "HeatingQC"] = data.loc[:, "HeatingQC"].fillna("TA")
    data.loc[:, "KitchenAbvGr"] = data.loc[:, "KitchenAbvGr"].fillna(0)
    data.loc[:, "KitchenQual"] = data.loc[:, "KitchenQual"].fillna("TA")
    data.loc[:, "LotFrontage"] = data.loc[:, "LotFrontage"].fillna(0)
    data.loc[:, "LotShape"] = data.loc[:, "LotShape"].fillna("Reg")
    data.loc[:, "MasVnrType"] = data.loc[:, "MasVnrType"].fillna("None")
    data.loc[:, "MasVnrArea"] = data.loc[:, "MasVnrArea"].fillna(0)
    data.loc[:, "MiscFeature"] = data.loc[:, "MiscFeature"].fillna("No")
    data.loc[:, "MiscVal"] = data.loc[:, "MiscVal"].fillna(0)
    data.loc[:, "OpenPorchSF"] = data.loc[:, "OpenPorchSF"].fillna(0)
    data.loc[:, "PavedDrive"] = data.loc[:, "PavedDrive"].fillna("N")
    data.loc[:, "PoolQC"] = data.loc[:, "PoolQC"].fillna("No")
    data.loc[:, "PoolArea"] = data.loc[:, "PoolArea"].fillna(0)
    data.loc[:, "SaleCondition"] = data.loc[:, "SaleCondition"].fillna("Normal")
    data.loc[:, "ScreenPorch"] = data.loc[:, "ScreenPorch"].fillna(0)
    data.loc[:, "TotRmsAbvGrd"] = data.loc[:, "TotRmsAbvGrd"].fillna(0)
    data.loc[:, "Utilities"] = data.loc[:, "Utilities"].fillna("AllPub")
    data.loc[:, "WoodDeckSF"] = data.loc[:, "WoodDeckSF"].fillna(0)

    data = data.replace({"MSSubClass": {20: "SC20", 30: "SC30", 40: "SC40", 45: "SC45",
                                          50: "SC50", 60: "SC60", 70: "SC70", 75: "SC75",
                                          80: "SC80", 85: "SC85", 90: "SC90", 120: "SC120",
                                          150: "SC150", 160: "SC160", 180: "SC180", 190: "SC190"},
                           "MoSold": {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                                      7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
                           })

    data = data.replace(int_to_cat)

    categorical_features = data.select_dtypes(include=["object"]).columns
    numerical_features = data.select_dtypes(exclude=["object"]).columns
    numerical_features = numerical_features.drop("SalePrice")
    X_num = data[numerical_features]
    X_cat = data[categorical_features]
    X_num = X_num.fillna(X_num.median())
    X_cat = pd.get_dummies(X_cat)
    X = pd.concat([X_num, X_cat], axis=1)

    return X, y


def transform_to_readable(X, y):
    print(X.head())


def transform_for_model(X, y):
    std_sc = joblib.load(standardizer_filename)
    numerical_features = X.select_dtypes(exclude=["object"]).columns
    X_std = std_sc.transform(X.loc[:, numerical_features])
    y_log = np.log1p(y)
    return X_std, y_log
