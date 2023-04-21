import numpy as np
from shap import LinearExplainer

from pyreal.explainers import GlobalFeatureImportance, ShapFeatureImportance


def test_fit_shap(all_models):
    for model in all_models:
        gfi_object = GlobalFeatureImportance(
            model=model["model"],
            x_train_orig=model["x"],
            transformers=model["transformers"],
            e_algorithm="shap",
            classes=np.arange(1, 4),
        )
        gfi_object.fit()
        shap = ShapFeatureImportance(
            model=model["model"],
            x_train_orig=model["x"],
            transformers=model["transformers"],
            classes=np.arange(1, 4),
        )
        shap.fit()

        assert shap.explainer is not None
        assert isinstance(shap.explainer, LinearExplainer)


def test_produce_shap_regression_no_transforms(regression_no_transforms):
    model = regression_no_transforms
    gfi = GlobalFeatureImportance(
        model=model["model"],
        x_train_orig=model["x"],
        e_algorithm="shap",
        transformers=model["transformers"],
        fit_on_init=True,
    )
    shap = ShapFeatureImportance(
        model=model["model"],
        x_train_orig=model["x"],
        transformers=model["transformers"],
        fit_on_init=True,
    )

    helper_produce_shap_regression_no_transforms(gfi, model)
    helper_produce_shap_regression_no_transforms(shap, model)


def helper_produce_shap_regression_no_transforms(explainer, model):
    importances = explainer.produce().get()
    assert importances.shape == (1, model["x"].shape[1])
    assert abs(importances["A"][0] - (4 / 3)) < 0.0001
    assert abs(importances["B"][0]) < 0.0001
    assert abs(importances["C"][0]) < 0.0001


def test_produce_shap_regression_transforms(regression_one_hot):
    model = regression_one_hot
    gfi = GlobalFeatureImportance(
        model=model["model"],
        x_train_orig=model["x"],
        e_algorithm="shap",
        transformers=model["transformers"],
        fit_on_init=True,
    )
    shap = ShapFeatureImportance(
        model=model["model"],
        x_train_orig=model["x"],
        transformers=model["transformers"],
        fit_on_init=True,
    )

    helper_produce_shap_regression_one_hot(gfi, regression_one_hot)
    helper_produce_shap_regression_one_hot(shap, regression_one_hot)


def helper_produce_shap_regression_one_hot(explainer, model):
    importances = explainer.produce().get()
    assert importances.shape == (1, model["x"].shape[1])
    assert abs(importances["A"][0] - (8 / 3)) < 0.0001
    assert abs(importances["B"][0]) < 0.0001
    assert abs(importances["C"][0]) < 0.0001


def test_shap_produce_classification_no_transforms(classification_no_transforms):
    model = classification_no_transforms
    gfi = GlobalFeatureImportance(
        model=model["model"],
        x_train_orig=model["x"],
        e_algorithm="shap",
        transformers=model["transformers"],
        fit_on_init=True,
        classes=np.arange(1, 4),
    )
    shap = ShapFeatureImportance(
        model=model["model"],
        x_train_orig=model["x"],
        transformers=model["transformers"],
        fit_on_init=True,
        classes=np.arange(1, 4),
    )

    helper_shap_produce_classification_no_transforms(gfi, classification_no_transforms)
    helper_shap_produce_classification_no_transforms(shap, classification_no_transforms)


def helper_shap_produce_classification_no_transforms(explainer, model):
    importances = explainer.produce().get()
    assert importances.shape == (1, model["x"].shape[1])
    assert abs(importances["A"][0]) < 0.0001
    assert abs(importances["B"][0] - 1) < 0.0001
    assert abs(importances["C"][0] - (2 / 3)) < 0.0001


def test_produce_shap_regression_transforms_with_size(regression_one_hot):
    model = regression_one_hot
    gfi = GlobalFeatureImportance(
        model=model["model"],
        x_train_orig=model["x"],
        e_algorithm="shap",
        transformers=model["transformers"],
        training_size=2,
        fit_on_init=True,
    )
    shap = ShapFeatureImportance(
        model=model["model"],
        x_train_orig=model["x"],
        transformers=model["transformers"],
        fit_on_init=True,
    )

    helper_produce_shap_regression_one_hot_with_size(gfi, regression_one_hot)
    helper_produce_shap_regression_one_hot_with_size(shap, regression_one_hot)


def helper_produce_shap_regression_one_hot_with_size(explainer, model):
    importances = explainer.produce().get()
    assert importances.shape == (1, model["x"].shape[1])


def test_shap_produce_classification_no_transforms_with_size(classification_no_transforms):
    model = classification_no_transforms
    gfi = GlobalFeatureImportance(
        model=model["model"],
        x_train_orig=model["x"],
        e_algorithm="shap",
        transformers=model["transformers"],
        fit_on_init=True,
        training_size=4,
        classes=np.arange(1, 4),
    )
    shap = ShapFeatureImportance(
        model=model["model"],
        x_train_orig=model["x"],
        transformers=model["transformers"],
        fit_on_init=True,
        training_size=4,
        classes=np.arange(1, 4),
    )

    helper_shap_produce_classification_no_transforms_with_size(gfi, classification_no_transforms)
    helper_shap_produce_classification_no_transforms_with_size(shap, classification_no_transforms)


def helper_shap_produce_classification_no_transforms_with_size(explainer, model):
    importances = explainer.produce().get()
    assert importances.shape == (1, model["x"].shape[1])
    assert abs(importances["C"][0]) > 0.0001
