import os
import pickle

import pandas as pd
from sklearn.linear_model import Ridge

from pyreal import RealApp
from pyreal.transformers import OneHotEncoder, Transformer, fit_transformers, run_transformers

DATA_DIR = os.path.join(os.path.dirname(__file__), "data_ames_housing")
DATA_FILE = os.path.join(DATA_DIR, "data.csv")
MODEL_FILE = os.path.join(DATA_DIR, "model.pkl")
DESCRIPTION_FILE = os.path.join(DATA_DIR, "feature_descriptions.csv")
TRANSFORMER_FILE = os.path.join(DATA_DIR, "transformers.pkl")


class AmesHousingImputer(Transformer):
    def data_transform(self, x):
        x_transform = x.copy()
        # Alley : data description says NA means "no alley access"
        x_transform.loc[:, "Alley"] = x_transform.loc[:, "Alley"].fillna("None")
        # BedroomAbvGr : NA most likely means 0
        x_transform.loc[:, "BedroomAbvGr"] = x_transform.loc[:, "BedroomAbvGr"].fillna(0)
        # BsmtQual etc : x_transform description says NA for basement features is "no basement"
        x_transform.loc[:, "BsmtQual"] = x_transform.loc[:, "BsmtQual"].fillna("No")
        x_transform.loc[:, "BsmtCond"] = x_transform.loc[:, "BsmtCond"].fillna("No")
        x_transform.loc[:, "BsmtExposure"] = x_transform.loc[:, "BsmtExposure"].fillna("No")
        x_transform.loc[:, "BsmtFinType1"] = x_transform.loc[:, "BsmtFinType1"].fillna("No")
        x_transform.loc[:, "BsmtFinType2"] = x_transform.loc[:, "BsmtFinType2"].fillna("No")
        x_transform.loc[:, "BsmtFullBath"] = x_transform.loc[:, "BsmtFullBath"].fillna(0)
        x_transform.loc[:, "BsmtHalfBath"] = x_transform.loc[:, "BsmtHalfBath"].fillna(0)
        x_transform.loc[:, "BsmtUnfSF"] = x_transform.loc[:, "BsmtUnfSF"].fillna(0)
        # CentralAir : NA most likely means No
        x_transform.loc[:, "CentralAir"] = x_transform.loc[:, "CentralAir"].fillna("N")
        # Condition : NA most likely means Normal
        x_transform.loc[:, "Condition1"] = x_transform.loc[:, "Condition1"].fillna("Norm")
        x_transform.loc[:, "Condition2"] = x_transform.loc[:, "Condition2"].fillna("Norm")
        # EnclosedPorch : NA most likely means no enclosed porch
        x_transform.loc[:, "EnclosedPorch"] = x_transform.loc[:, "EnclosedPorch"].fillna(0)
        # External stuff : NA most likely means average
        x_transform.loc[:, "ExterCond"] = x_transform.loc[:, "ExterCond"].fillna("TA")
        x_transform.loc[:, "ExterQual"] = x_transform.loc[:, "ExterQual"].fillna("TA")
        # Fence : x_transform description says NA means "no fence"
        x_transform.loc[:, "Fence"] = x_transform.loc[:, "Fence"].fillna("No")
        # FireplaceQu : x_transform description says NA means "no fireplace"
        x_transform.loc[:, "FireplaceQu"] = x_transform.loc[:, "FireplaceQu"].fillna("No")
        x_transform.loc[:, "Fireplaces"] = x_transform.loc[:, "Fireplaces"].fillna(0)
        # Functional : x_transform description says NA means typical
        x_transform.loc[:, "Functional"] = x_transform.loc[:, "Functional"].fillna("Typ")
        # GarageType etc : x_transform description says NA for garage features is "no garage"
        x_transform.loc[:, "GarageType"] = x_transform.loc[:, "GarageType"].fillna("No")
        x_transform.loc[:, "GarageFinish"] = x_transform.loc[:, "GarageFinish"].fillna("No")
        x_transform.loc[:, "GarageQual"] = x_transform.loc[:, "GarageQual"].fillna("No")
        x_transform.loc[:, "GarageCond"] = x_transform.loc[:, "GarageCond"].fillna("No")
        x_transform.loc[:, "GarageArea"] = x_transform.loc[:, "GarageArea"].fillna(0)
        x_transform.loc[:, "GarageCars"] = x_transform.loc[:, "GarageCars"].fillna(0)
        # HalfBath : NA most likely means no half baths above grade
        x_transform.loc[:, "HalfBath"] = x_transform.loc[:, "HalfBath"].fillna(0)
        # HeatingQC : NA most likely means typical
        x_transform.loc[:, "HeatingQC"] = x_transform.loc[:, "HeatingQC"].fillna("TA")
        # KitchenAbvGr : NA most likely means 0
        x_transform.loc[:, "KitchenAbvGr"] = x_transform.loc[:, "KitchenAbvGr"].fillna(0)
        # KitchenQual : NA most likely means typical
        x_transform.loc[:, "KitchenQual"] = x_transform.loc[:, "KitchenQual"].fillna("TA")
        # LotFrontage : NA most likely means no lot frontage
        x_transform.loc[:, "LotFrontage"] = x_transform.loc[:, "LotFrontage"].fillna(0)
        # LotShape : NA most likely means regular
        x_transform.loc[:, "LotShape"] = x_transform.loc[:, "LotShape"].fillna("Reg")
        # MasVnrType : NA most likely means no veneer
        x_transform.loc[:, "MasVnrType"] = x_transform.loc[:, "MasVnrType"].fillna("None")
        x_transform.loc[:, "MasVnrArea"] = x_transform.loc[:, "MasVnrArea"].fillna(0)
        # MiscFeature : x_transform description says NA means "no misc feature"
        x_transform.loc[:, "MiscFeature"] = x_transform.loc[:, "MiscFeature"].fillna("No")
        x_transform.loc[:, "MiscVal"] = x_transform.loc[:, "MiscVal"].fillna(0)
        # OpenPorchSF : NA most likely means no open porch
        x_transform.loc[:, "OpenPorchSF"] = x_transform.loc[:, "OpenPorchSF"].fillna(0)
        # PavedDrive : NA most likely means not paved
        x_transform.loc[:, "PavedDrive"] = x_transform.loc[:, "PavedDrive"].fillna("N")
        # PoolQC : x_transform description says NA means "no pool"
        x_transform.loc[:, "PoolQC"] = x_transform.loc[:, "PoolQC"].fillna("No")
        x_transform.loc[:, "PoolArea"] = x_transform.loc[:, "PoolArea"].fillna(0)
        # SaleCondition : NA most likely means normal sale
        x_transform.loc[:, "SaleCondition"] = x_transform.loc[:, "SaleCondition"].fillna("Normal")
        # ScreenPorch : NA most likely means no screen porch
        x_transform.loc[:, "ScreenPorch"] = x_transform.loc[:, "ScreenPorch"].fillna(0)
        # TotRmsAbvGrd : NA most likely means 0
        x_transform.loc[:, "TotRmsAbvGrd"] = x_transform.loc[:, "TotRmsAbvGrd"].fillna(0)
        # Utilities : NA most likely means all public utilities
        x_transform.loc[:, "Utilities"] = x_transform.loc[:, "Utilities"].fillna("AllPub")
        # WoodDeckSF : NA most likely means no wood deck
        x_transform.loc[:, "WoodDeckSF"] = x_transform.loc[:, "WoodDeckSF"].fillna(0)
        x_transform.loc[:, "Electrical"] = x_transform.loc[:, "Electrical"].fillna("SBrkr")

        x_num = x_transform.select_dtypes(exclude=["object"])
        x_cat = x_transform.select_dtypes(include=["object"])
        x_num = x_num.fillna(x_num.median())

        return pd.concat([x_num, x_cat], axis=1)


def load_feature_descriptions():
    descriptions_df = pd.read_csv(DESCRIPTION_FILE)
    return descriptions_df.set_index("Name").to_dict()["Description"]


def load_data(n_rows=None, include_targets=False):
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
    else:
        raise FileNotFoundError("Ames housing data is missing")

    df = df[df.GrLivArea < 4000]
    y = df["SalePrice"]
    x_orig = df.drop("SalePrice", axis="columns")

    if n_rows is not None and include_targets:
        return x_orig[:n_rows], y[:n_rows]
    elif n_rows is not None:
        return x_orig[:n_rows]
    elif include_targets:
        return x_orig, y
    else:
        return x_orig


def load_model():
    if os.path.exists(MODEL_FILE):
        return pickle.load(open(os.path.join(DATA_DIR, "model.pkl"), "rb"))
    else:
        transformers = load_transformers()
        x_orig, y = load_data(include_targets=True)
        x_orig = x_orig.drop("Id", axis="columns")
        x_model = run_transformers(transformers, x_orig)
        model = Ridge()
        model.fit(x_model, y)

        if not os.path.isdir(DATA_DIR):
            os.mkdir(DATA_DIR)
        with open(MODEL_FILE, "wb") as f:
            pickle.dump(model, f)
        return model


def load_transformers():
    if os.path.exists(TRANSFORMER_FILE):
        return pickle.load(open(os.path.join(DATA_DIR, "transformers.pkl"), "rb"))
    else:
        x_orig, y = load_data(include_targets=True)
        x_orig = x_orig.drop("Id", axis="columns")
        ames_imputer = AmesHousingImputer()
        x_imputed = fit_transformers(ames_imputer, x_orig)
        object_columns = x_imputed.select_dtypes(include=["object"]).columns
        onehotencoder = OneHotEncoder(object_columns)
        fit_transformers(onehotencoder, x_imputed)

        transformers = [ames_imputer, onehotencoder]

        if not os.path.isdir(DATA_DIR):
            os.mkdir(DATA_DIR)
        with open(TRANSFORMER_FILE, "wb") as f:
            pickle.dump(transformers, f)
        return transformers


def load_app():
    x_train_orig, y = load_data(include_targets=True)
    model = load_model()
    transformers = load_transformers()
    feature_descriptions = load_feature_descriptions()

    return RealApp(
        model,
        x_train_orig,
        y_train=y,
        transformers=transformers,
        feature_descriptions=feature_descriptions,
        id_column="Id",
    )
