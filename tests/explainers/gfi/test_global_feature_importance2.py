import numpy as np
import pandas as pd
from shap import LinearExplainer

from pyreal.explainers import GlobalFeatureImportance, ShapFeatureImportance


def test_fit_shap(all_models):
    for model in all_models:
        gfi_object = GlobalFeatureImportance(
            model=model["model"],
            x_train_orig=model["x"], transforms=model["transforms"],
            e_algorithm='shap')
        gfi_object.fit()
        shap = ShapFeatureImportance(
            model=model["model"],
            x_train_orig=model["x"], transforms=model["transforms"])
        shap.fit()
        assert shap.explainer is not None
        assert isinstance(shap.explainer, LinearExplainer)


def test_produce_shap_regression_no_transforms(regression_no_transforms):
    model = regression_no_transforms
    gfi = GlobalFeatureImportance(model=model["model"],
                                  x_train_orig=model["x"], e_algorithm='shap',
                                  transforms=model["transforms"],
                                  fit_on_init=True)
    shap = ShapFeatureImportance(
        model=model["model"], x_train_orig=model["x"], transforms=model["transforms"],
        fit_on_init=True)

    helper_produce_shap_regression_no_transforms(gfi, model)
    helper_produce_shap_regression_no_transforms(shap, model)


def helper_produce_shap_regression_no_transforms(explainer, model):
    importances = explainer.produce()
    assert importances.shape == (1, model["x"].shape[1])
    assert abs(importances["A"][0] - (4 / 3)) < 0.0001
    assert abs(importances["B"][0]) < 0.0001
    assert abs(importances["C"][0]) < 0.0001


def test_produce_shap_regression_transforms(regression_one_hot):
    model = regression_one_hot
    gfi = GlobalFeatureImportance(model=model["model"],
                                  x_train_orig=model["x"], e_algorithm='shap',
                                  transforms=model["transforms"],
                                  fit_on_init=True)
    shap = ShapFeatureImportance(
        model=model["model"], x_train_orig=model["x"], transforms=model["transforms"],
        fit_on_init=True)

    helper_produce_shap_regression_one_hot(gfi, regression_one_hot)
    helper_produce_shap_regression_one_hot(shap, regression_one_hot)


def helper_produce_shap_regression_one_hot(explainer, model):
    importances = explainer.produce()
    assert importances.shape == (1, model["x"].shape[1])
    assert abs(importances["A"][0] - (8/3)) < .0001
    assert abs(importances["B"][0]) < .0001
    assert abs(importances["C"][0]) < .0001


def test_produce_shap_classification_no_transforms(classification_no_transforms):
    model = classification_no_transforms
    gfi = GlobalFeatureImportance(model=model["model"],
                                  x_train_orig=model["x"], e_algorithm='shap',
                                  transforms=model["transforms"],
                                  fit_on_init=True,
                                  classes=np.arange(1, 4))
    shap = ShapFeatureImportance(
        model=model["model"], x_train_orig=model["x"], transforms=model["transforms"],
        fit_on_init=True, classes=np.arange(1, 4))

    helper_produce_shap_classification_no_transforms(gfi, classification_no_transforms)
    helper_produce_shap_classification_no_transforms(shap, classification_no_transforms)


def helper_produce_shap_classification_no_transforms(explainer, model):
    importances = explainer.produce()
    assert importances.shape == (1, model["x"].shape[1])
    assert abs(importances["A"][0] - (2/3)) < .0001
    assert abs(importances["B"][0] - (2/3)) < .0001
    assert abs(importances["C"][0] - (2/3)) < .0001


def test_produce_with_renames(regression_one_hot):
    model = regression_one_hot
    e_transforms = model["transforms"]
    feature_descriptions = {"A": "Feature A", "B": "Feature B"}
    gfi = GlobalFeatureImportance(model=model["model"],
                                  x_train_orig=model["x"], e_algorithm='shap',
                                  fit_on_init=True, e_transforms=e_transforms,
                                  interpretable_features=True,
                                  feature_descriptions=feature_descriptions)

    importances = gfi.produce()
    assert importances.shape == (1, model["x"].shape[1])
    assert abs(importances["Feature A"][0] - (8/3)) < 0.0001
    assert abs(importances["Feature B"][0]) < 0.0001
    assert abs(importances["C"][0]) < 0.0001
