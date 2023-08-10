import matplotlib
import pandas as pd

from pyreal import RealApp
from pyreal.explainers import GlobalFeatureImportance, LocalFeatureContribution
from pyreal.visualize import feature_bar_plot, swarm_plot


def test_feature_bar_plot_lfc_no_break(regression_no_transforms):
    matplotlib.use("Agg")

    realApp = RealApp(
        regression_no_transforms["model"],
        regression_no_transforms["x"],
        transformers=regression_no_transforms["transformers"],
    )

    x_one_dim = pd.DataFrame([[2, 10, 10]], columns=["A", "B", "C"])
    explanation = realApp.produce_feature_contributions(x_one_dim)

    feature_bar_plot(explanation[next(iter(explanation))], show=False)


def test_feature_bar_plot_gfi_no_break(regression_no_transforms):
    matplotlib.use("Agg")

    realApp = RealApp(
        regression_no_transforms["model"],
        regression_no_transforms["x"],
        transformers=regression_no_transforms["transformers"],
    )

    explanation = realApp.produce_feature_importance()

    feature_bar_plot(explanation, show=False)


def test_feature_bar_plot_lfc_object_no_break(regression_no_transforms):
    matplotlib.use("Agg")

    lfc = LocalFeatureContribution(
        model=regression_no_transforms["model"],
        x_train_orig=regression_no_transforms["x"],
        e_algorithm="shap",
        transformers=regression_no_transforms["transformers"],
        fit_on_init=True,
    )

    x_one_dim = pd.DataFrame([[2, 10, 10]], columns=["A", "B", "C"])
    explanation = lfc.produce(x_one_dim)
    feature_bar_plot(explanation, show=False)


def test_feature_bar_plot_gfi_object_no_break(regression_no_transforms):
    matplotlib.use("Agg")

    lfc = GlobalFeatureImportance(
        model=regression_no_transforms["model"],
        x_train_orig=regression_no_transforms["x"],
        e_algorithm="shap",
        transformers=regression_no_transforms["transformers"],
        fit_on_init=True,
    )

    explanation = lfc.produce()
    feature_bar_plot(explanation, show=False)


def test_plot_swarm_lfc_no_break(regression_no_transforms):
    matplotlib.use("Agg")

    realApp = RealApp(
        regression_no_transforms["model"],
        regression_no_transforms["x"],
        transformers=regression_no_transforms["transformers"],
    )

    x_multi_dim = pd.DataFrame(
        [[2, 10, 10], [2, 10, 10], [2, 2, 2], [1, 1, 1]], columns=["A", "B", "C"]
    )
    explanation = realApp.produce_feature_contributions(x_multi_dim)

    swarm_plot(explanation, show=False)


def test_swarm_plot_lfc_object_no_break(regression_no_transforms):
    matplotlib.use("Agg")

    lfc = LocalFeatureContribution(
        model=regression_no_transforms["model"],
        x_train_orig=regression_no_transforms["x"],
        e_algorithm="shap",
        transformers=regression_no_transforms["transformers"],
        fit_on_init=True,
    )

    x_multi_dim = pd.DataFrame(
        [[2, 10, 10], [2, 10, 10], [2, 2, 2], [1, 1, 1]], columns=["A", "B", "C"]
    )
    explanation = lfc.produce(x_multi_dim)
    swarm_plot(explanation, show=False)
