import numpy as np
import pandas as pd
from shap import LinearExplainer

from pyreal.explainers import LocalFeatureContribution, ShapFeatureContribution


def test_fit_shap(all_models):
    for model in all_models:
        lfc_object = LocalFeatureContribution(
            model=model["model"],
            x_train_orig=model["x"], transforms=model["transforms"],
            e_algorithm='shap')
        lfc_object.fit()
        shap = ShapFeatureContribution(
            model=model["model"],
            x_train_orig=model["x"], transforms=model["transforms"])
        shap.fit()
        assert shap.explainer is not None
        assert isinstance(shap.explainer, LinearExplainer)


def test_produce_shap_regression_no_transforms(regression_no_transforms):
    model = regression_no_transforms
    lfc = LocalFeatureContribution(model=model["model"],
                                   x_train_orig=model["x"], e_algorithm='shap',
                                   transforms=model["transforms"],
                                   fit_on_init=True)
    shap = ShapFeatureContribution(
        model=model["model"], x_train_orig=model["x"], transforms=model["transforms"],
        fit_on_init=True)

    helper_produce_shap_regression_no_transforms(lfc, model)
    helper_produce_shap_regression_no_transforms(shap, model)


def helper_produce_shap_regression_no_transforms(explainer, model):
    x_one_dim = pd.DataFrame([[2, 10, 10]], columns=["A", "B", "C"])
    x_multi_dim = pd.DataFrame([[2, 1, 1],
                                [4, 2, 3]], columns=["A", "B", "C"])
    expected = np.mean(model["y"])[0]
    contributions = explainer.produce(x_one_dim)
    assert x_one_dim.shape == contributions.shape
    assert contributions.iloc[0, 0] == x_one_dim.iloc[0, 0] - expected
    assert contributions.iloc[0, 1] == 0
    assert contributions.iloc[0, 2] == 0

    contributions = explainer.produce(x_multi_dim)
    assert x_multi_dim.shape == contributions.shape
    assert contributions.iloc[0, 0] == x_multi_dim.iloc[0, 0] - expected
    assert contributions.iloc[1, 0] == x_multi_dim.iloc[1, 0] - expected
    assert (contributions.iloc[:, 1] == 0).all()
    assert (contributions.iloc[:, 2] == 0).all()


def test_produce_shap_regression_transforms(regression_one_hot):
    model = regression_one_hot
    lfc = LocalFeatureContribution(model=model["model"],
                                   x_train_orig=model["x"], e_algorithm='shap',
                                   transforms=model["transforms"],
                                   fit_on_init=True)
    shap = ShapFeatureContribution(
        model=model["model"], x_train_orig=model["x"], transforms=model["transforms"],
        fit_on_init=True)

    helper_produce_shap_regression_one_hot(lfc)
    helper_produce_shap_regression_one_hot(shap)


def helper_produce_shap_regression_one_hot(explainer):
    x_one_dim = pd.DataFrame([[2, 10, 10]], columns=["A", "B", "C"])
    x_multi_dim = pd.DataFrame([[4, 1, 1],
                                [6, 2, 3]], columns=["A", "B", "C"])
    contributions = explainer.produce(x_one_dim)
    assert x_one_dim.shape == contributions.shape
    assert abs(contributions["A"][0] + 1) < .0001
    assert abs(contributions["B"][0]) < .0001
    assert abs(contributions["C"][0]) < .0001

    contributions = explainer.produce(x_multi_dim)
    assert x_multi_dim.shape == contributions.shape
    assert abs(contributions["A"][0]) < .0001
    assert abs(contributions["A"][0]) < .0001
    assert abs(contributions["A"][1] - 1 < .0001)
    assert (contributions["B"] == 0).all()
    assert (contributions["C"] == 0).all()


def test_produce_shap_classification_no_transforms(classification_no_transforms):
    model = classification_no_transforms
    lfc = LocalFeatureContribution(model=model["model"],
                                   x_train_orig=model["x"], e_algorithm='shap',
                                   transforms=model["transforms"],
                                   fit_on_init=True,
                                   classes=np.arange(1, 4))
    shap = ShapFeatureContribution(
        model=model["model"], x_train_orig=model["x"], transforms=model["transforms"],
        fit_on_init=True, classes=np.arange(1, 4))

    helper_produce_shap_classification_no_transforms(lfc)
    helper_produce_shap_classification_no_transforms(shap)


def helper_produce_shap_classification_no_transforms(explainer):
    x_one_dim = pd.DataFrame([[1, 0, 0]], columns=["A", "B", "C"])
    x_multi_dim = pd.DataFrame([[1, 0, 0],
                                [1, 0, 0]], columns=["A", "B", "C"])
    contributions = explainer.produce(x_one_dim)
    assert x_one_dim.shape == contributions.shape
    assert abs(contributions["A"][0]) < .0001
    assert abs(contributions["B"][0]) < .0001
    assert abs(contributions["C"][0]) < .0001

    contributions = explainer.produce(x_multi_dim)
    print(contributions)
    assert x_multi_dim.shape == contributions.shape
    assert abs(contributions["A"][0]) < .0001
    assert abs(contributions["A"][1] - 1 < .0001)
    assert (contributions["B"] == 0).all()
    assert (contributions["C"] == 0).all()


def test_produce_with_renames(regression_one_hot):
    model = regression_one_hot
    transforms = model["transforms"]
    feature_descriptions = {"A": "Feature A", "B": "Feature B"}
    lfc = LocalFeatureContribution(model=model["model"],
                                   x_train_orig=model["x"], e_algorithm='shap',
                                   fit_on_init=True, transforms=transforms,
                                   interpretable_features=True,
                                   feature_descriptions=feature_descriptions)
    x_one_dim = pd.DataFrame([[2, 10, 10]], columns=["A", "B", "C"])

    contributions = lfc.produce(x_one_dim)
    assert x_one_dim.shape == contributions.shape
    assert abs(contributions["Feature A"][0] + 1) < 0.0001
    assert abs(contributions["Feature B"][0]) < 0.0001
    assert abs(contributions["C"][0]) < 0.0001
