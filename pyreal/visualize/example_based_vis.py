import pandas as pd


def example_table(
    explanation, input_name="Original Input", y_column_name="Ground Truth", y_format_func=None
):
    table = explanation["X"]
    explanation["Input"].name = input_name
    table = pd.concat([explanation["Input"].to_frame().T, table])
    explanation["y"] = explanation["y"].squeeze()
    if y_format_func is not None:
        explanation["y"] = explanation["y"].apply(y_format_func)
    explanation["y"][input_name] = "N/A"
    table.insert(0, y_column_name, explanation["y"])
    return table
