from pyreal.transformers.time_series_formatter import (
    check_input,
    df_to_numpy2d,
    df_to_numpy3d,
    numpy2d_to_df,
    numpy3d_to_df,
)


def test_check_input(time_series_data):
    np3d = time_series_data["np3d"]
    np2d = time_series_data["np2d"]
    df3d = time_series_data["df3d"]
    df2d = time_series_data["df2d"]

    check_input(np3d)
    check_input(np2d)
    check_input(df3d)
    check_input(df2d)


def test_df_to_numpy2d(time_series_data):
    np2d = time_series_data["np2d"]
    df2d = time_series_data["df2d"]
    array = df_to_numpy2d(df2d)
    assert (array == np2d).all()


def test_df_to_numpy3d(time_series_data):
    np3d = time_series_data["np3d"]
    df3d = time_series_data["df3d"]
    array = df_to_numpy3d(df3d)
    assert (array == np3d).all()


def test_numpy2d_to_df(time_series_data):
    np2d = time_series_data["np2d"]
    df2d = time_series_data["df2d"]
    df = numpy2d_to_df(np2d, var_name="var_0")
    assert df.equals(df2d)


def test_numpy3d_to_df(time_series_data):
    np3d = time_series_data["np3d"]
    df3d = time_series_data["df3d"]
    df = numpy3d_to_df(np3d)
    assert df.equals(df3d)
