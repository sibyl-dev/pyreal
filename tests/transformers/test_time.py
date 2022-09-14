from pyreal.transformers import (
    MultiIndexFrameToNestedFrame,
    MultiIndexFrameToNumpy2d,
    MultiIndexFrameToNumpy3d,
    NestedFrameToMultiIndexFrame,
    NestedFrameToNumpy3d,
    Numpy2dToMultiIndexFrame,
    Numpy3dToMultiIndexFrame,
    Numpy3dToNestedFrame,
)


def test_df_to_sktime(time_series_data):
    df3d = time_series_data["df3d"]
    nested = time_series_data["nested"]
    transformer = MultiIndexFrameToNestedFrame()
    array = transformer.fit_transform(df3d)
    for col in nested:
        for i, series in enumerate(nested[col]):
            if not series.equals(array[col][i]):
                assert False


def test_df_to_numpy2d(time_series_data):
    np2d = time_series_data["np2d"]
    df2d = time_series_data["df2d"]
    transformer = MultiIndexFrameToNumpy2d()
    array = transformer.fit_transform(df2d)
    assert (array == np2d).all()


def test_df_to_numpy3d(time_series_data):
    np3d = time_series_data["np3d"]
    df3d = time_series_data["df3d"]
    transformer = MultiIndexFrameToNumpy3d()
    array = transformer.fit_transform(df3d)
    assert (array == np3d).all()


def test_numpy2d_to_df(time_series_data):
    np2d = time_series_data["np2d"]
    df2d = time_series_data["df2d"]
    transformer = Numpy2dToMultiIndexFrame()
    df = transformer.fit_transform(np2d)
    assert df.equals(df2d)


def test_numpy3d_to_df(time_series_data):
    np3d = time_series_data["np3d"]
    df3d = time_series_data["df3d"]
    transformer = Numpy3dToMultiIndexFrame()
    df = transformer.fit_transform(np3d)
    assert df.equals(df3d)


def test_numpy3d_to_sktime(time_series_data):
    np3d = time_series_data["np3d"]
    nested = time_series_data["nested"]
    transformer = Numpy3dToNestedFrame()
    array = transformer.fit_transform(np3d)
    for col in nested:
        for i, series in enumerate(nested[col]):
            if not series.equals(array[col][i]):
                assert False


def test_sktime_to_df(time_series_data):
    nested = time_series_data["nested"]
    df3d = time_series_data["df3d"]
    transformer = NestedFrameToMultiIndexFrame()
    df = transformer.fit_transform(nested)
    assert df.equals(df3d)


def test_sktime_to_numpy3d(time_series_data):
    nested = time_series_data["nested"]
    np3d = time_series_data["np3d"]
    transformer = NestedFrameToNumpy3d()
    array = transformer.fit_transform(nested)
    assert (array == np3d).all()
