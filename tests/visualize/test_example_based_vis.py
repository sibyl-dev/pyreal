import pandas as pd

from pyreal import RealApp
from pyreal.visualize import example_table, plot_explanation


def test_example_table_no_break(regression_no_transforms):
    real_app = RealApp(
        regression_no_transforms["model"],
        regression_no_transforms["x"],
        regression_no_transforms["y"],
        transformers=regression_no_transforms["transformers"],
    )

    x_one_dim = pd.DataFrame([[2, 10, 10]], columns=["A", "B", "C"])
    explanation = real_app.produce_similar_examples(x_one_dim)

    example_table(explanation[next(iter(explanation))])
    plot_explanation(explanation[next(iter(explanation))])
