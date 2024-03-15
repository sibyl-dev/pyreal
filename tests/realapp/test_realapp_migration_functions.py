import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder

from pyreal import RealApp
from pyreal.transformers import ColumnDropTransformer, OneHotEncoder, Transformer


class DummyTransformer:
    def transform(self, X):
        return X


def test_realapp_from_sklearn_model_only():
    model = LinearRegression()
    X = pd.DataFrame([[1, 2], [3, 4]], columns=["a", "b"])
    y = pd.Series([1, 2])
    realapp = RealApp.from_sklearn(model=model, X_train=X, y_train=y)
    assert realapp.models[0] is model
    assert realapp.X_train_orig is X
    assert realapp.y_train is y


def test_realapp_from_sklearn_model_and_transformers():
    model = LinearRegression()
    transformers = [SklearnOneHotEncoder(), DummyTransformer()]

    X = pd.DataFrame([[1, 2], [3, 4]], columns=["a", "b"])
    y = pd.Series([1, 2])
    realapp = RealApp.from_sklearn(model=model, transformers=transformers, X_train=X, y_train=y)
    assert realapp.models[0] is model
    assert realapp.X_train_orig is X
    assert realapp.y_train is y
    # check is realapp.transformers contains a OneHotEncoder and a Transformer
    assert len(realapp.transformers) == 2
    assert isinstance(realapp.transformers[0], OneHotEncoder)
    assert isinstance(realapp.transformers[1], Transformer)
    assert realapp.transformers[1].wrapped_transformer is transformers[1]


def test_realapp_from_sklearn_pipeline():
    model = LinearRegression()
    dummy = DummyTransformer()
    pipeline = Pipeline([("onehot", SklearnOneHotEncoder()), ("dummy", dummy), ("model", model)])

    X = pd.DataFrame([[1, 2], [3, 4]], columns=["a", "b"])
    y = pd.Series([1, 2])
    realapp = RealApp.from_sklearn(pipeline=pipeline, X_train=X, y_train=y)
    assert realapp.models[0] is model
    assert realapp.X_train_orig is X
    assert realapp.y_train is y
    # check realapp.transformers contains a OneHotEncoder and a Transformer
    assert len(realapp.transformers) == 2
    assert isinstance(realapp.transformers[0], OneHotEncoder)
    assert isinstance(realapp.transformers[1], Transformer)
    assert realapp.transformers[1].wrapped_transformer is dummy


@pytest.mark.parametrize("fitted", [True, False])
def test_realapp_from_sklearn_pipeline_with_column_transformer(fitted):
    model = LinearRegression()
    column_transformer = ColumnTransformer(
        [("onehot", SklearnOneHotEncoder(), ["a"]), ("drop", "drop", ["b"])]
    )
    pipeline = Pipeline([("column_transformer", column_transformer), ("model", model)])

    X = pd.DataFrame([[1, 2], [3, 4]], columns=["a", "b"])
    y = pd.Series([1, 2])
    if fitted:
        pipeline.fit(X, y)
    realapp = RealApp.from_sklearn(pipeline=pipeline, X_train=X, y_train=y)
    assert realapp.models[0] is model
    assert realapp.X_train_orig is X
    assert realapp.y_train is y
    assert len(realapp.transformers) == 2
    assert isinstance(realapp.transformers[0], OneHotEncoder)
    assert realapp.transformers[0].columns == ["a"]
    assert isinstance(realapp.transformers[1], ColumnDropTransformer)
    assert realapp.transformers[1].dropped_columns == ["b"]
