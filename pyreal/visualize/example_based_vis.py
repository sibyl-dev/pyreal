import pandas as pd


def example_table(
    explanation, input_name="Original Input", y_column_name="Ground Truth", y_format_func=None
):
    table = explanation["X"].copy()
    y = explanation["y"].copy()
    input_col = explanation["Input"].copy()
    input_col.name = input_name
    y.index = explanation["X"].index
    table = pd.concat([input_col.to_frame().T, table])
    y = explanation["y"].squeeze()
    if y_format_func is not None:
        y = y.apply(y_format_func)
    y[input_name] = "N/A"
    table.insert(0, y_column_name, y)
    return table
