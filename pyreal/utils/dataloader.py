"""
Utility functions for loading time-series data
"""
import pandas as pd

# from scipy.io import arff


def format_csv(df, timestamp_column=None, value_columns=None):
    timestamp_column_name = df.columns[timestamp_column] if timestamp_column else df.columns[0]
    value_column_names = df.columns[value_columns] if value_columns else df.columns[1:]

    data = dict()
    data["timestamp"] = df[timestamp_column_name].astype("int64").values
    for column in value_column_names:
        data[column] = df[column].astype(float).values

    return pd.DataFrame(data)


def load_csv(path, timestamp_column=None, value_column=None):
    header = None if timestamp_column is not None else "infer"
    data = pd.read_csv(path, header=header)

    if timestamp_column is None:
        if value_column is not None:
            raise ValueError("If value_column is provided, timestamp_column must be as well")

        return data

    elif value_column is None:
        raise ValueError("If timestamp_column is provided, value_column must be as well")
    elif timestamp_column == value_column:
        raise ValueError("timestamp_column cannot be the same as value_column")

    return format_csv(data, timestamp_column, value_column)


# def load_arff(path):
#     """
#     Load .arff format files into Pyreal time_series format.

#     Args:
#         path (.arff filepath):
#             Path to input data

#     Returns:
#         DataFrame of (n_instances, [n_columns, n_timestamps])
#     """
#     data, meta = arff.loadarff(path)
#     X = data[...,:-1]
#     y = data[...,-1]
#     return X, y


# def load_ts(path):
#     """
#     Load .ts format files into Pyreal time_series format.

#     Args:
#         path (.ts filepath):
#             Path to input data

#     Returns:
#         DataFrame of (n_instances, [n_columns, n_timestamps])
#     """
#     data, meta = arff.loadarff(path)
#     X = data[..., :-1]
#     y = data[..., -1]
#     return X, y
