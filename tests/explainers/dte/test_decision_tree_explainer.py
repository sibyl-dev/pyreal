import numpy as np
from sklearn.tree import plot_tree
from sklearn.tree import export_text

from pyreal.explainers import DecisionTreeExplainer, SurrogateDecisionTree


def test_produce_decision_tree_regression_no_transforms(regression_no_transforms):
    model = regression_no_transforms
    dte = DecisionTreeExplainer(model=model["model"],
                                x_train_orig=model["x"],
                                e_algorithm='surrogate_tree',
                                transforms=model["transforms"],
                                fit_on_init=True)
    SUdte = SurrogateDecisionTree(
        model=model["model"], x_train_orig=model["x"], transforms=model["transforms"],
        fit_on_init=True)

    helper_produce_decision_tree_regression_no_transforms(dte, model)
    helper_produce_decision_tree_regression_no_transforms(SUdte, model)


def helper_produce_decision_tree_regression_no_transforms(explainer, model):
    tree_object = explainer.produce()
    print(export_text(tree_object))
    # assert importances.shape == (1, model["x"].shape[1])
    # assert abs(importances["A"][0] - (4 / 3)) < 0.0001
    # assert abs(importances["B"][0]) < 0.0001
    # assert abs(importances["C"][0]) < 0.0001


def test_produce_decision_tree_regression_transforms(regression_one_hot):
    model = regression_one_hot
    dte = DecisionTreeExplainer(model=model["model"],
                                x_train_orig=model["x"],
                                e_algorithm='surrogate_tree',
                                transforms=model["transforms"],
                                fit_on_init=True)
    SUdte = SurrogateDecisionTree(
        model=model["model"], x_train_orig=model["x"], transforms=model["transforms"],
        fit_on_init=True)

    helper_produce_decision_tree_regression_one_hot(dte, model)
    helper_produce_decision_tree_regression_one_hot(SUdte, model)


def helper_produce_decision_tree_regression_one_hot(explainer, model):
    tree_object = explainer.produce()
    plot_tree(tree_object)
    # assert importances.shape == (1, model["x"].shape[1])
    # assert abs(importances["A"][0] - (8 / 3)) < .0001
    # assert abs(importances["B"][0]) < .0001
    # assert abs(importances["C"][0]) < .0001


def test_produce_decision_tree_classification_no_transforms(classification_no_transforms):
    model = classification_no_transforms
    dte = DecisionTreeExplainer(model=model["model"],
                                x_train_orig=model["x"],
                                e_algorithm='surrogate_tree',
                                transforms=model["transforms"],
                                fit_on_init=True,
                                classes=np.arange(1, 4))
    SUdte = SurrogateDecisionTree(
        model=model["model"], x_train_orig=model["x"], transforms=model["transforms"],
        fit_on_init=True, classes=np.arange(1, 4))

    helper_produce_decision_tree_classification_no_transforms(
        dte, classification_no_transforms)
    helper_produce_decision_tree_classification_no_transforms(
        SUdte, classification_no_transforms)


def helper_produce_decision_tree_classification_no_transforms(explainer, model):
    tree_object = explainer.produce()
    plot_tree(tree_object)
    # assert importances.shape == (1, model["x"].shape[1])
    # assert abs(importances["A"][0] - (2 / 3)) < .0001
    # assert abs(importances["B"][0] - (2 / 3)) < .0001
    # assert abs(importances["C"][0] - (2 / 3)) < .0001


def test_produce_with_renames(regression_one_hot):
    model = regression_one_hot
    e_transforms = model["transforms"]
    feature_descriptions = {"A": "Feature A", "B": "Feature B"}
    dte = DecisionTreeExplainer(model=model["model"],
                                x_train_orig=model["x"],
                                is_classifier=True,
                                e_algorithm='surrogate_tree',
                                fit_on_init=True, e_transforms=e_transforms,
                                interpretable_features=True,
                                feature_descriptions=feature_descriptions)

    tree_object = dte.produce()
    plot_tree(tree_object)
    # assert importances.shape == (1, model["x"].shape[1])
    # assert abs(importances["Feature A"][0] - (8 / 3)) < 0.0001
    # assert abs(importances["Feature B"][0]) < 0.0001
    # assert abs(importances["C"][0]) < 0.0001
