import numpy as np

from pyreal.explainers import GlobalFeatureImportance, PermutationFeatureImportance

def test_produce_permutation_regression_no_transforms(regression_no_transforms):
    model = regression_no_transforms
    gfi = GlobalFeatureImportance(
        model=model["model"],
        x_train_orig=model["x"],
        y_orig=model["y"],
        e_algorithm="permutation",
        transformers=model["transformers"],
        fit_on_init=True,
    )
    shap = PermutationFeatureImportance(
        model=model["model"],
        x_train_orig=model["x"],
        y_orig=model["y"],
        transformers=model["transformers"],
        fit_on_init=True,
    )

    helper_produce_permutation_regression_no_transforms(gfi, model)
    helper_produce_permutation_regression_no_transforms(shap, model)


def helper_produce_permutation_regression_no_transforms(explainer, model):
    importances = explainer.produce()
    assert importances.shape == (1, model["x"].shape[1])
    assert importances["A"][0] > 0.0001
    assert abs(importances["B"][0]) < 0.0001
    assert abs(importances["C"][0]) < 0.0001


def test_produce_permutation_regression_transforms(regression_one_hot):
    model = regression_one_hot
    gfi = GlobalFeatureImportance(
        model=model["model"],
        x_train_orig=model["x"],
        y_orig=model["y"],
        e_algorithm="permutation",
        transformers=model["transformers"],
        fit_on_init=True,
    )
    shap = PermutationFeatureImportance(
        model=model["model"],
        x_train_orig=model["x"],
        y_orig=model["y"],
        transformers=model["transformers"],
        fit_on_init=True,
    )

    helper_produce_permutation_regression_one_hot(gfi, regression_one_hot)
    helper_produce_permutation_regression_one_hot(shap, regression_one_hot)


def helper_produce_permutation_regression_one_hot(explainer, model):
    importances = explainer.produce()
    assert importances.shape == (1, 5)
    assert importances["A_2"][0] > 0.0001
    assert importances["A_4"][0] > 0.0001
    assert importances["A_6"][0] > 0.0001
    assert abs(importances["B"][0]) < 0.0001
    assert abs(importances["C"][0]) < 0.0001


def test_permutation_produce_classification_no_transforms(classification_no_transforms):
    model = classification_no_transforms
    gfi = GlobalFeatureImportance(
        model=model["model"],
        x_train_orig=model["x"],
        y_orig=model["y"],
        e_algorithm="permutation",
        transformers=model["transformers"],
        fit_on_init=True,
        classes=np.arange(1, 4),
    )
    permutation = PermutationFeatureImportance(
        model=model["model"],
        x_train_orig=model["x"],
        y_orig=model["y"],
        transformers=model["transformers"],
        fit_on_init=True,
        classes=np.arange(1, 4),
    )

    helper_permutation_produce_classification_no_transforms(gfi, classification_no_transforms)
    helper_permutation_produce_classification_no_transforms(
        permutation, classification_no_transforms
    )


def helper_permutation_produce_classification_no_transforms(explainer, model):
    importances = explainer.produce()
    assert importances.shape == (1, model["x"].shape[1])
    assert abs(importances["A"][0]) < 0.0001
    assert importances["B"][0] > 0.0001
    assert importances["C"][0] > 0.0001


def test_produce_permutation_regression_no_transforms_with_size(regression_no_transforms):
    model = regression_no_transforms
    gfi = GlobalFeatureImportance(
        model=model["model"],
        x_train_orig=model["x"],
        y_orig=model["y"],
        e_algorithm="permutation",
        transformers=model["transformers"],
        fit_on_init=True,
        training_size=4,
    )
    shap = PermutationFeatureImportance(
        model=model["model"],
        x_train_orig=model["x"],
        y_orig=model["y"],
        transformers=model["transformers"],
        fit_on_init=True,
        training_size=4,
    )

    helper_produce_permutation_regression_no_transforms_with_size(gfi, model)
    helper_produce_permutation_regression_no_transforms_with_size(shap, model)


def helper_produce_permutation_regression_no_transforms_with_size(explainer, model):
    importances = explainer.produce()
    assert importances.shape == (1, model["x"].shape[1])


def test_permutation_produce_classification_no_transforms_with_size(classification_no_transforms):
    model = classification_no_transforms
    gfi = GlobalFeatureImportance(
        model=model["model"],
        x_train_orig=model["x"],
        y_orig=model["y"],
        e_algorithm="permutation",
        transformers=model["transformers"],
        fit_on_init=True,
        training_size=5,
        classes=np.arange(1, 4),
    )
    permutation = PermutationFeatureImportance(
        model=model["model"],
        x_train_orig=model["x"],
        y_orig=model["y"],
        transformers=model["transformers"],
        fit_on_init=True,
        training_size=5,
        classes=np.arange(1, 4),
    )

    helper_permutation_produce_with_size(gfi, classification_no_transforms)
    helper_permutation_produce_with_size(permutation, classification_no_transforms)


def helper_permutation_produce_with_size(explainer, model):
    importances = explainer.produce()
    assert importances.shape == (1, model["x"].shape[1])
