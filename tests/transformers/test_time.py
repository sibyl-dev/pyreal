from pyreal.transformers import (
    df_to_nested,
    df_to_np2d,
    df_to_np3d,
    nested_to_df,
    nested_to_np3d,
    np2d_to_df,
    np3d_to_df,
)


def test_df_to_sktime(time_series_data):
    df3d = time_series_data["df3d"]
    nested = time_series_data["nested"]
    transformer = df_to_nested()
    array = transformer.fit_transform(df3d)
    assert (array == nested).all()


def test_df_to_numpy2d(time_series_data):
    np2d = time_series_data["np2d"]
    df2d = time_series_data["df2d"]
    transformer = df_to_np2d()
    array = transformer.fit_transform(df2d)
    assert (array == np2d).all()


def test_df_to_numpy3d(time_series_data):
    np3d = time_series_data["np3d"]
    df3d = time_series_data["df3d"]
    transformer = df_to_np3d()
    array = transformer.fit_transform(df3d)
    assert (array == np3d).all()


def test_numpy2d_to_df(time_series_data):
    np2d = time_series_data["np2d"]
    df2d = time_series_data["df2d"]
    transformer = np2d_to_df()
    df = transformer.fit_transform(np2d)
    assert df.equals(df2d)


def test_numpy3d_to_df(time_series_data):
    np3d = time_series_data["np3d"]
    df3d = time_series_data["df3d"]
    transformer = np3d_to_df()
    df = transformer.fit_transform(np3d)
    assert df.equals(df3d)


def test_sktime_to_df(time_series_data):
    nested = time_series_data["nested"]
    df3d = time_series_data["df3d"]
    transformer = nested_to_df()
    df = transformer.fit_transform(nested)
    assert df.equals(df3d)


def test_sktime_to_numpy3d(time_series_data):
    nested = time_series_data["nested"]
    np3d = time_series_data["np3d"]
    transformer = nested_to_np3d()
    array = transformer.fit_transform(nested)
    assert (array == np3d).all()
