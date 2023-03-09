from pyreal import RealApp
import pandas as pd
from pyreal.visualize import plot_top_contributors
from pyreal.explainers import LocalFeatureContribution, GlobalFeatureImportance


def test_plot_top_contributors_lfc_no_break(regression_no_transforms):
    realApp = RealApp(
        regression_no_transforms["model"],
        regression_no_transforms["x"],
        transformers=regression_no_transforms["transformers"],
    )

    x_one_dim = pd.DataFrame([[2, 10, 10]], columns=["A", "B", "C"])
    explanation = realApp.produce_local_feature_contributions(x_one_dim)

    plot_top_contributors(explanation[next(iter(explanation))], show=False)


def test_plot_top_contributors_gfi_no_break(regression_no_transforms):
    realApp = RealApp(
        regression_no_transforms["model"],
        regression_no_transforms["x"],
        transformers=regression_no_transforms["transformers"],
    )

    explanation = realApp.produce_global_feature_importance()

    plot_top_contributors(explanation, show=False)


def test_plot_top_contributors_lfc_object_no_break(regression_no_transforms):
    lfc = LocalFeatureContribution(
        model=regression_no_transforms["model"],
        x_train_orig=regression_no_transforms["x"],
        e_algorithm="shap",
        transformers=regression_no_transforms["transformers"],
        fit_on_init=True,
    )

    x_one_dim = pd.DataFrame([[2, 10, 10]], columns=["A", "B", "C"])
    explanation = lfc.produce(x_one_dim)
    plot_top_contributors(explanation, show=False)


def test_plot_top_contributors_gfi_object_no_break(regression_no_transforms):
    lfc = GlobalFeatureImportance(
        model=regression_no_transforms["model"],
        x_train_orig=regression_no_transforms["x"],
        e_algorithm="shap",
        transformers=regression_no_transforms["transformers"],
        fit_on_init=True,
    )

    explanation = lfc.produce()
    plot_top_contributors(explanation, show=False)
