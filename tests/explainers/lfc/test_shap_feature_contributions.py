import numpy as np
import pandas as pd
from shap import LinearExplainer

import pyreal.explainers.lfc.shap_feature_contribution
from pyreal.explainers import LocalFeatureContribution, ShapFeatureContribution


def test_average_or_mode():
    mix = pd.DataFrame([[1, "a", 2, "a"], [1, "a", 4, "a"], [1, "b", 3, "a"]])
    expected = [1, "a", 3, "a"]
    result = pyreal.explainers.lfc.shap_feature_contribution._get_average_or_mode(mix)
    for i in range(len(expected)):
        assert expected[i] == result[i]

    mode_only = pd.DataFrame([["a", "a", "b", "d"], ["b", "a", "b", "d"], ["a", "b", "d", "d"]])
    expected = ["a", "a", "b", "d"]
    result = pyreal.explainers.lfc.shap_feature_contribution._get_average_or_mode(mode_only)
    for i in range(len(expected)):
        assert expected[i] == result[i]

    mean_only = pd.DataFrame([[1, 0, 2, 2], [1, 5, 3, 3], [1, -5, 4, 4]])
    expected = [1, 0, 3, 3]
    result = pyreal.explainers.lfc.shap_feature_contribution._get_average_or_mode(mean_only)
    for i in range(len(expected)):
        assert expected[i] == result[i]


def test_fit_shap(all_models):
    for model in all_models:
        lfc_object = LocalFeatureContribution(
            model=model["model"],
            x_train_orig=model["x"],
            transformers=model["transformers"],
            e_algorithm="shap",
        )
        lfc_object.fit()
        shap = ShapFeatureContribution(
            model=model["model"], x_train_orig=model["x"], transformers=model["transformers"]
        )
        shap.fit()

        assert shap.explainer is not None
        assert isinstance(shap.explainer, LinearExplainer)


def test_produce_shap_regression_no_transforms(regression_no_transforms):
    model = regression_no_transforms
    lfc = LocalFeatureContribution(
        model=model["model"],
        x_train_orig=model["x"],
        e_algorithm="shap",
        transformers=model["transformers"],
        fit_on_init=True,
    )
    shap = ShapFeatureContribution(
        model=model["model"],
        x_train_orig=model["x"],
        transformers=model["transformers"],
        fit_on_init=True,
    )
    averages = list(model["x"].mean(axis="rows"))
    helper_produce_shap_regression_no_transforms(lfc, model, averages=averages)
    helper_produce_shap_regression_no_transforms(shap, model, averages=averages)


def helper_produce_shap_regression_no_transforms(explainer, model, averages):
    x_one_dim = pd.DataFrame([[2, 10, 10]], columns=["A", "B", "C"])
    x_multi_dim = pd.DataFrame([[2, 1, 1], [4, 2, 3]], columns=["A", "B", "C"])
    expected = np.mean(model["y"])
    explanation = explainer.produce(x_one_dim)
    contributions = explanation.get()
    assert x_one_dim.shape == contributions.shape
    assert contributions.iloc[0, 0] == x_one_dim.iloc[0, 0] - expected
    assert contributions.iloc[0, 1] == 0
    assert contributions.iloc[0, 2] == 0
    assert list(explanation.get_average_values()) == averages

    explanation = explainer.produce(x_multi_dim)
    contributions = explanation.get()
    assert x_multi_dim.shape == contributions.shape
    assert contributions.iloc[0, 0] == x_multi_dim.iloc[0, 0] - expected
    assert contributions.iloc[1, 0] == x_multi_dim.iloc[1, 0] - expected
    assert (contributions.iloc[:, 1] == 0).all()
    assert (contributions.iloc[:, 2] == 0).all()
    assert list(explanation.get_average_values()) == averages


def test_produce_shap_regression_with_index_names(regression_no_transforms):
    explainer = ShapFeatureContribution(
        model=regression_no_transforms["model"],
        x_train_orig=regression_no_transforms["x"],
        transformers=regression_no_transforms["transformers"],
        fit_on_init=True,
    )

    x_multi_dim = pd.DataFrame(
        [[2, 1, 1], [4, 2, 3]], columns=["A", "B", "C"], index=["row1", "row2"]
    )
    expected = np.mean(regression_no_transforms["y"])

    contributions = explainer.produce(x_multi_dim).get()
    assert x_multi_dim.shape == contributions.shape
    assert contributions.loc["row1", "A"] == x_multi_dim.loc["row1", "A"] - expected
    assert contributions.iloc[1, 0] == x_multi_dim.iloc[1, 0] - expected
    assert (contributions.iloc[:, 1] == 0).all()
    assert (contributions.iloc[:, 2] == 0).all()


def test_produce_shap_regression_transforms(regression_one_hot):
    model = regression_one_hot
    lfc = LocalFeatureContribution(
        model=model["model"],
        x_train_orig=model["x"],
        e_algorithm="shap",
        transformers=model["transformers"],
        fit_on_init=True,
    )
    shap = ShapFeatureContribution(
        model=model["model"],
        x_train_orig=model["x"],
        transformers=model["transformers"],
        fit_on_init=True,
    )
    averages = list(model["x"].mean(axis="rows"))
    helper_produce_shap_regression_one_hot(lfc, averages=averages)
    helper_produce_shap_regression_one_hot(shap, averages=averages)


def test_produce_shap_no_dataset_on_init(regression_one_hot):
    model = regression_one_hot
    x = model["x"]
    lfc = LocalFeatureContribution(
        model=model["model"],
        e_algorithm="shap",
        transformers=model["transformers"],
    )
    shap = ShapFeatureContribution(
        model=model["model"],
        transformers=model["transformers"],
    )
    lfc.fit(x)
    shap.fit(x)

    averages = list(model["x"].mean(axis="rows"))
    helper_produce_shap_regression_one_hot(lfc, averages)
    helper_produce_shap_regression_one_hot(shap, averages)


def helper_produce_shap_regression_one_hot(explainer, averages):
    x_one_dim = pd.DataFrame([[2, 10, 10]], columns=["A", "B", "C"])
    x_multi_dim = pd.DataFrame([[4, 1, 1], [6, 2, 3]], columns=["A", "B", "C"])
    explanation = explainer.produce(x_one_dim)
    contributions = explanation.get()
    assert x_one_dim.shape == contributions.shape
    assert abs(contributions["A"][0] + 1) < 0.0001
    assert abs(contributions["B"][0]) < 0.0001
    assert abs(contributions["C"][0]) < 0.0001
    assert list(explanation.get_average_values()) == averages

    explanation = explainer.produce(x_multi_dim)
    contributions = explanation.get()
    assert x_multi_dim.shape == contributions.shape
    assert abs(contributions["A"][0]) < 0.0001
    assert abs(contributions["A"][1] - 1 < 0.0001)
    assert (contributions["B"] == 0).all()
    assert (contributions["C"] == 0).all()
    assert list(explanation.get_average_values()) == averages


def test_produce_shap_classification_no_transforms(classification_no_transforms):
    model = classification_no_transforms
    lfc = LocalFeatureContribution(
        model=model["model"],
        x_train_orig=model["x"],
        e_algorithm="shap",
        transformers=model["transformers"],
        fit_on_init=True,
        classes=np.arange(1, 4),
    )
    shap = ShapFeatureContribution(
        model=model["model"],
        x_train_orig=model["x"],
        transformers=model["transformers"],
        fit_on_init=True,
        classes=np.arange(1, 4),
    )

    helper_produce_shap_classification_no_transforms(lfc)
    helper_produce_shap_classification_no_transforms(shap)


def helper_produce_shap_classification_no_transforms(explainer):
    x_one_dim = pd.DataFrame([[1, 0, 0]], columns=["A", "B", "C"])
    x_multi_dim = pd.DataFrame([[1, 0, 0], [1, 1, 0]], columns=["A", "B", "C"])
    contributions = explainer.produce(x_one_dim).get()
    assert x_one_dim.shape == contributions.shape
    assert abs(contributions["A"][0]) < 0.0001
    assert abs(contributions["B"][0] + 1) < 0.0001
    assert abs(contributions["C"][0]) < 0.0001

    contributions = explainer.produce(x_multi_dim).get()
    assert x_multi_dim.shape == contributions.shape
    assert (contributions["A"] == 0).all()
    assert abs(contributions["B"][0] + 1) < 0.0001
    assert abs(contributions["B"][1]) < 0.0001
    assert (contributions["C"] == 0).all()


def test_fit_shap_with_size(all_models):
    for model in all_models:
        shap_with_size = ShapFeatureContribution(
            model=model["model"],
            x_train_orig=model["x"],
            transformers=model["transformers"],
            training_size=2,
        )
        shap_with_size.fit()

        assert shap_with_size.explainer is not None
        assert isinstance(shap_with_size.explainer, LinearExplainer)


def test_produce_shap_regression_no_transforms_with_size(regression_no_transforms):
    model = regression_no_transforms

    shap = ShapFeatureContribution(
        model=model["model"],
        x_train_orig=model["x"],
        transformers=model["transformers"],
        fit_on_init=True,
        training_size=2,
    )

    helper_produce_shap_regression_no_transforms_with_size(shap, model)


def helper_produce_shap_regression_no_transforms_with_size(explainer, model):
    x_one_dim = pd.DataFrame([[2, 10, 10]], columns=["A", "B", "C"])
    x_multi_dim = pd.DataFrame([[2, 1, 1], [4, 2, 3]], columns=["A", "B", "C"])

    contributions = explainer.produce(x_one_dim).get()
    assert x_one_dim.shape == contributions.shape
    contributions = explainer.produce(x_multi_dim).get()
    assert x_multi_dim.shape == contributions.shape
    assert (contributions.iloc[:, 1] == 0).all()
    assert (contributions.iloc[:, 2] == 0).all()


def test_produce_shap_regression_transforms_with_size(regression_one_hot):
    model = regression_one_hot

    shap = ShapFeatureContribution(
        model=model["model"],
        x_train_orig=model["x"],
        transformers=model["transformers"],
        fit_on_init=True,
        training_size=2,
    )

    helper_produce_shap_regression_one_hot_with_size(shap)


def helper_produce_shap_regression_one_hot_with_size(explainer):
    x_one_dim = pd.DataFrame([[2, 10, 10]], columns=["A", "B", "C"])
    x_multi_dim = pd.DataFrame([[4, 1, 1], [6, 2, 3]], columns=["A", "B", "C"])
    contributions = explainer.produce(x_one_dim).get()
    assert x_one_dim.shape == contributions.shape
    assert abs(contributions["B"][0]) < 0.0001
    assert abs(contributions["C"][0]) < 0.0001

    contributions = explainer.produce(x_multi_dim).get()
    assert x_multi_dim.shape == contributions.shape
    assert (contributions["B"] == 0).all()
    assert (contributions["C"] == 0).all()


def test_produce_shap_classification_no_transforms_with_size(classification_no_transforms):
    model = classification_no_transforms
    shap = ShapFeatureContribution(
        model=model["model"],
        x_train_orig=model["x"],
        transformers=model["transformers"],
        fit_on_init=True,
        classes=np.arange(1, 4),
        training_size=3,
    )

    helper_produce_shap_classification_no_transforms_with_size(shap)


def helper_produce_shap_classification_no_transforms_with_size(explainer):
    x_one_dim = pd.DataFrame([[1, 0, 0]], columns=["A", "B", "C"])
    x_multi_dim = pd.DataFrame([[1, 0, 0], [1, 0, 0]], columns=["A", "B", "C"])
    contributions = explainer.produce(x_one_dim).get()
    assert x_one_dim.shape == contributions.shape

    contributions = explainer.produce(x_multi_dim).get()
    assert x_multi_dim.shape == contributions.shape
