import numpy as np

from pyreal.explainers import DecisionTreeExplainer


def test_produce_with_renames(classification_no_transform_tree):
    model = classification_no_transform_tree
    transforms = model["transformers"]
    feature_descriptions = {"A": "Feature A", "B": "Feature B"}
    dte = DecisionTreeExplainer(
        model=model["model"],
        x_train_orig=model["x"],
        is_classifier=True,
        e_algorithm="surrogate_tree",
        fit_on_init=True,
        transformers=transforms,
        interpretable_features=True,
        feature_descriptions=feature_descriptions,
    )

    tree_object = dte.produce()
    assert tree_object.feature_importances_.shape == (
        dte.transform_to_x_algorithm(model["x"]).shape[1],
    )
    assert (tree_object.predict(model["x"].to_numpy()) == model["y"].to_numpy().ravel()).all()


def test_produce_with_renames_with_size(classification_no_transform_tree):
    model = classification_no_transform_tree
    transforms = model["transformers"]
    feature_descriptions = {"A": "Feature A", "B": "Feature B"}
    dte = DecisionTreeExplainer(
        model=model["model"],
        x_train_orig=model["x"],
        is_classifier=True,
        e_algorithm="surrogate_tree",
        fit_on_init=True,
        transformers=transforms,
        interpretable_features=True,
        feature_descriptions=feature_descriptions,
        training_size=3,
    )

    tree_object = dte.produce()
    assert tree_object.feature_importances_.shape == (
        dte.transform_to_x_algorithm(model["x"]).shape[1],
    )
