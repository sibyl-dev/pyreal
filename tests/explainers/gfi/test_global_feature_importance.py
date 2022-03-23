import numpy as np
from shap import LinearExplainer

from pyreal.explainers import (
    GlobalFeatureImportance,
    PermutationFeatureImportance,
    ShapFeatureImportance,
)


def test_fit_shap(all_models):
    for model in all_models:
        gfi_object = GlobalFeatureImportance(
            model=model["model"],
            x_train_orig=model["x"], transformers=model["transformers"],
            e_algorithm='shap')
        gfi_object.fit()
        shap = ShapFeatureImportance(
            model=model["model"],
            x_train_orig=model["x"], transformers=model["transformers"])
        shap.fit()

        assert shap.explainer is not None
        assert isinstance(shap.explainer, LinearExplainer)


def test_produce_shap_regression_no_transforms(regression_no_transforms):
    model = regression_no_transforms
    gfi = GlobalFeatureImportance(model=model["model"],
                                  x_train_orig=model["x"], e_algorithm='shap',
                                  transformers=model["transformers"],
                                  fit_on_init=True)
    shap = ShapFeatureImportance(
        model=model["model"], x_train_orig=model["x"], transformers=model["transformers"],
        fit_on_init=True)

    helper_produce_shap_regression_no_transforms(gfi, model)
    helper_produce_shap_regression_no_transforms(shap, model)


def helper_produce_shap_regression_no_transforms(explainer, model):
    importances = explainer.produce()
    assert importances.shape == (1, model["x"].shape[1])
    assert abs(importances["A"][0] - (4 / 3)) < 0.0001
    assert abs(importances["B"][0]) < 0.0001
    assert abs(importances["C"][0]) < 0.0001


def test_produce_permutation_regression_no_transforms(regression_no_transforms):
    model = regression_no_transforms
    gfi = GlobalFeatureImportance(model=model["model"],
                                  x_train_orig=model["x"], y_orig=model["y"],
                                  e_algorithm='permutation',
                                  transformers=model["transformers"],
                                  fit_on_init=True)
    shap = PermutationFeatureImportance(
        model=model["model"], x_train_orig=model["x"], y_orig=model["y"],
        transformers=model["transformers"],
        fit_on_init=True)

    helper_produce_permutation_regression_no_transforms(gfi, model)
    helper_produce_permutation_regression_no_transforms(shap, model)


def helper_produce_permutation_regression_no_transforms(explainer, model):
    importances = explainer.produce()
    assert importances.shape == (1, model["x"].shape[1])
    assert importances["A"][0] > 0.0001
    assert abs(importances["B"][0]) < 0.0001
    assert abs(importances["C"][0]) < 0.0001


def test_produce_shap_regression_transforms(regression_one_hot):
    model = regression_one_hot
    gfi = GlobalFeatureImportance(model=model["model"],
                                  x_train_orig=model["x"], e_algorithm='shap',
                                  transformers=model["transformers"],
                                  fit_on_init=True)
    shap = ShapFeatureImportance(
        model=model["model"], x_train_orig=model["x"], transformers=model["transformers"],
        fit_on_init=True)

    helper_produce_shap_regression_one_hot(gfi, regression_one_hot)
    helper_produce_shap_regression_one_hot(shap, regression_one_hot)


def helper_produce_shap_regression_one_hot(explainer, model):
    importances = explainer.produce()
    assert importances.shape == (1, model["x"].shape[1])
    assert abs(importances["A"][0] - (8 / 3)) < .0001
    assert abs(importances["B"][0]) < .0001
    assert abs(importances["C"][0]) < .0001


def test_produce_permutation_regression_transforms(regression_one_hot):
    model = regression_one_hot
    gfi = GlobalFeatureImportance(model=model["model"],
                                  x_train_orig=model["x"], y_orig=model["y"],
                                  e_algorithm='permutation',
                                  transformers=model["transformers"],
                                  fit_on_init=True)
    shap = PermutationFeatureImportance(
        model=model["model"], x_train_orig=model["x"], y_orig=model["y"],
        transformers=model["transformers"],
        fit_on_init=True)

    helper_produce_permutation_regression_one_hot(gfi, regression_one_hot)
    helper_produce_permutation_regression_one_hot(shap, regression_one_hot)


def helper_produce_permutation_regression_one_hot(explainer, model):
    importances = explainer.produce()
    assert importances.shape == (1, model["x"].shape[1])
    assert importances["A"][0] > .0001
    assert abs(importances["B"][0]) < .0001
    assert abs(importances["C"][0]) < .0001


def test_shap_produce_classification_no_transforms(classification_no_transforms):
    model = classification_no_transforms
    gfi = GlobalFeatureImportance(model=model["model"],
                                  x_train_orig=model["x"], e_algorithm='shap',
                                  transformers=model["transformers"],
                                  fit_on_init=True,
                                  classes=np.arange(1, 4))
    shap = ShapFeatureImportance(
        model=model["model"], x_train_orig=model["x"], transformers=model["transformers"],
        fit_on_init=True, classes=np.arange(1, 4))

    helper_shap_produce_classification_no_transforms(gfi, classification_no_transforms)
    helper_shap_produce_classification_no_transforms(shap, classification_no_transforms)


def helper_shap_produce_classification_no_transforms(explainer, model):
    importances = explainer.produce()
    assert importances.shape == (1, model["x"].shape[1])
    assert abs(importances["A"][0]) < .0001
    assert abs(importances["B"][0] - 1) < .0001
    assert abs(importances["C"][0] - (2 / 3)) < .0001


def test_permutation_produce_classification_no_transforms(classification_no_transforms):
    model = classification_no_transforms
    gfi = GlobalFeatureImportance(model=model["model"],
                                  x_train_orig=model["x"], y_orig=model["y"],
                                  e_algorithm='permutation',
                                  transformers=model["transformers"],
                                  fit_on_init=True,
                                  classes=np.arange(1, 4))
    permutation = PermutationFeatureImportance(
        model=model["model"], x_train_orig=model["x"], y_orig=model["y"],
        transformers=model["transformers"], fit_on_init=True, classes=np.arange(1, 4))

    helper_permutation_produce_classification_no_transforms(gfi, classification_no_transforms)
    helper_permutation_produce_classification_no_transforms(permutation,
                                                            classification_no_transforms)


def helper_permutation_produce_classification_no_transforms(explainer, model):
    importances = explainer.produce()
    assert importances.shape == (1, model["x"].shape[1])
    assert abs(importances["A"][0]) < .0001
    assert importances["B"][0] > .0001
    assert importances["C"][0] > .0001


def test_produce_with_renames(regression_one_hot):
    model = regression_one_hot
    transforms = model["transformers"]
    feature_descriptions = {"A": "Feature A", "B": "Feature B"}
    gfi = GlobalFeatureImportance(model=model["model"],
                                  x_train_orig=model["x"], e_algorithm='shap',
                                  fit_on_init=True, transformers=transforms,
                                  interpretable_features=True,
                                  feature_descriptions=feature_descriptions)

    importances = gfi.produce()
    assert importances.shape == (1, model["x"].shape[1])
    assert abs(importances["Feature A"][0] - (8 / 3)) < 0.0001
    assert abs(importances["Feature B"][0]) < 0.0001
    assert abs(importances["C"][0]) < 0.0001


def test_evaluate_variation(classification_no_transforms):
    model = classification_no_transforms
    lfc = GlobalFeatureImportance(model=model["model"],
                                  x_train_orig=model["x"], e_algorithm='shap',
                                  transformers=model["transformers"],
                                  fit_on_init=True,
                                  classes=np.arange(1, 4))

    # Assert no crash. Values analyzed through benchmarking
    lfc.evaluate_variation(with_fit=False, n_iterations=5)
    lfc.evaluate_variation(with_fit=True, n_iterations=5)


'''Tests below here test `training_size`'''


def test_produce_shap_regression_transforms_with_size(regression_one_hot):
    model = regression_one_hot
    gfi = GlobalFeatureImportance(model=model["model"],
                                  x_train_orig=model["x"], e_algorithm='shap',
                                  transformers=model["transformers"],
                                  training_size=2,
                                  fit_on_init=True)
    shap = ShapFeatureImportance(
        model=model["model"], x_train_orig=model["x"], transformers=model["transformers"],
        fit_on_init=True)

    helper_produce_shap_regression_one_hot_with_size(gfi, regression_one_hot)
    helper_produce_shap_regression_one_hot_with_size(shap, regression_one_hot)


def helper_produce_shap_regression_one_hot_with_size(explainer, model):
    importances = explainer.produce()
    assert importances.shape == (1, model["x"].shape[1])


def test_produce_permutation_regression_no_transforms_with_size(regression_no_transforms):
    model = regression_no_transforms
    gfi = GlobalFeatureImportance(model=model["model"],
                                  x_train_orig=model["x"], y_orig=model["y"],
                                  e_algorithm='permutation',
                                  transformers=model["transformers"],
                                  fit_on_init=True,
                                  training_size=4)
    shap = PermutationFeatureImportance(
        model=model["model"], x_train_orig=model["x"], y_orig=model["y"],
        transformers=model["transformers"],
        fit_on_init=True,
        training_size=4)

    helper_produce_permutation_regression_no_transforms_with_size(gfi, model)
    helper_produce_permutation_regression_no_transforms_with_size(shap, model)


def helper_produce_permutation_regression_no_transforms_with_size(explainer, model):
    importances = explainer.produce()
    assert importances.shape == (1, model["x"].shape[1])


def test_shap_produce_classification_no_transforms_with_size(classification_no_transforms):
    model = classification_no_transforms
    gfi = GlobalFeatureImportance(model=model["model"],
                                  x_train_orig=model["x"], e_algorithm='shap',
                                  transformers=model["transformers"],
                                  fit_on_init=True,
                                  training_size=4,
                                  classes=np.arange(1, 4))
    shap = ShapFeatureImportance(
        model=model["model"], x_train_orig=model["x"], transformers=model["transformers"],
        fit_on_init=True, training_size=4, classes=np.arange(1, 4))

    helper_shap_produce_classification_no_transforms_with_size(gfi, classification_no_transforms)
    helper_shap_produce_classification_no_transforms_with_size(shap, classification_no_transforms)


def helper_shap_produce_classification_no_transforms_with_size(explainer, model):
    importances = explainer.produce()
    assert importances.shape == (1, model["x"].shape[1])
    assert abs(importances["C"][0]) > .0001


def test_permutation_produce_classification_no_transforms_with_size(classification_no_transforms):
    model = classification_no_transforms
    gfi = GlobalFeatureImportance(model=model["model"],
                                  x_train_orig=model["x"], y_orig=model["y"],
                                  e_algorithm='permutation',
                                  transformers=model["transformers"],
                                  fit_on_init=True,
                                  training_size=5,
                                  classes=np.arange(1, 4))
    permutation = PermutationFeatureImportance(
        model=model["model"], x_train_orig=model["x"], y_orig=model["y"],
        transformers=model["transformers"], fit_on_init=True,
        training_size=5, classes=np.arange(1, 4))

    helper_permutation_produce_with_size(gfi, classification_no_transforms)
    helper_permutation_produce_with_size(permutation, classification_no_transforms)


def helper_permutation_produce_with_size(explainer, model):
    importances = explainer.produce()
    assert importances.shape == (1, model["x"].shape[1])


def test_evaluate_variation_with_size(classification_no_transforms):
    model = classification_no_transforms
    lfc = GlobalFeatureImportance(model=model["model"],
                                  x_train_orig=model["x"], e_algorithm='shap',
                                  transformers=model["transformers"],
                                  fit_on_init=True,
                                  training_size=5,
                                  classes=np.arange(1, 4))

    # Assert no crash. Values analyzed through benchmarking
    lfc.evaluate_variation(with_fit=False, n_iterations=5)
    lfc.evaluate_variation(with_fit=True, n_iterations=5)
