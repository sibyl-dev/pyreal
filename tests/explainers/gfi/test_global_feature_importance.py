import numpy as np

from pyreal.explainers import GlobalFeatureImportance


def test_produce_with_renames(regression_one_hot):
    model = regression_one_hot
    transforms = model["transformers"]
    feature_descriptions = {"A": "Feature A", "B": "Feature B"}
    gfi = GlobalFeatureImportance(
        model=model["model"],
        x_train_orig=model["x"],
        e_algorithm="shap",
        fit_on_init=True,
        transformers=transforms,
        feature_descriptions=feature_descriptions,
    )

    importances = gfi.produce().get()
    assert importances.shape == (1, model["x"].shape[1])
    assert abs(importances["Feature A"][0] - (8 / 3)) < 0.0001
    assert abs(importances["Feature B"][0]) < 0.0001
    assert abs(importances["C"][0]) < 0.0001


def test_produce_no_dataset_on_init(regression_one_hot):
    model = regression_one_hot
    x = model["x"]
    transforms = model["transformers"]
    feature_descriptions = {"A": "Feature A", "B": "Feature B"}
    gfi = GlobalFeatureImportance(
        model=model["model"],
        e_algorithm="shap",
        transformers=transforms,
        feature_descriptions=feature_descriptions,
    )
    gfi.fit(x)

    importances = gfi.produce().get()
    assert importances.shape == (1, x.shape[1])
    assert abs(importances["Feature A"][0] - (8 / 3)) < 0.0001
    assert abs(importances["Feature B"][0]) < 0.0001
    assert abs(importances["C"][0]) < 0.0001


def test_evaluate_variation(classification_no_transforms):
    model = classification_no_transforms
    gfi = GlobalFeatureImportance(
        model=model["model"],
        x_train_orig=model["x"],
        e_algorithm="shap",
        transformers=model["transformers"],
        fit_on_init=True,
        classes=np.arange(1, 4),
    )

    # Assert no crash. Values analyzed through benchmarking
    gfi.evaluate_variation(with_fit=False, n_iterations=5)
    gfi.evaluate_variation(with_fit=True, n_iterations=5)


def test_evaluate_variation_with_size(classification_no_transforms):
    model = classification_no_transforms
    gfi = GlobalFeatureImportance(
        model=model["model"],
        x_train_orig=model["x"],
        e_algorithm="shap",
        transformers=model["transformers"],
        fit_on_init=True,
        training_size=5,
        classes=np.arange(1, 4),
    )

    # Assert no crash. Values analyzed through benchmarking
    gfi.evaluate_variation(with_fit=False, n_iterations=5)
    gfi.evaluate_variation(with_fit=True, n_iterations=5)
