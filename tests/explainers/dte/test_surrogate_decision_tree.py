import numpy as np

from pyreal.explainers import DecisionTreeExplainer, SurrogateDecisionTree


def test_produce_decision_tree_regression_no_transforms(regression_no_transforms):
    model = regression_no_transforms
    dte = DecisionTreeExplainer(
        model=model["model"],
        x_train_orig=model["x"],
        e_algorithm="surrogate_tree",
        is_classifier=False,
        max_depth=5,
        transformers=model["transformers"],
        fit_on_init=True,
    )
    SUdte = SurrogateDecisionTree(
        model=model["model"],
        x_train_orig=model["x"],
        is_classifier=False,
        max_depth=5,
        transformers=model["transformers"],
        fit_on_init=True,
    )

    helper_produce_decision_tree_regression_no_transforms(dte, model)
    helper_produce_decision_tree_regression_no_transforms(SUdte, model)


def helper_produce_decision_tree_regression_no_transforms(explainer, model):
    tree_object = explainer.produce().get()
    assert tree_object.feature_importances_.shape == (
        explainer.transform_to_x_algorithm(model["x"]).shape[1],
    )


def test_produce_decision_tree_regression_transforms(regression_one_hot):
    model = regression_one_hot
    dte = DecisionTreeExplainer(
        model=model["model"],
        x_train_orig=model["x"],
        e_algorithm="surrogate_tree",
        is_classifier=False,
        transformers=model["transformers"],
        fit_on_init=True,
    )
    SUdte = SurrogateDecisionTree(
        model=model["model"],
        x_train_orig=model["x"],
        transformers=model["transformers"],
        fit_on_init=True,
    )

    helper_produce_decision_tree_regression_one_hot(dte, model)
    helper_produce_decision_tree_regression_one_hot(SUdte, model)


def test_produce_decision_tree_no_dataset_on_init(regression_one_hot):
    model = regression_one_hot
    x = model["x"]
    dte = DecisionTreeExplainer(
        model=model["model"],
        e_algorithm="surrogate_tree",
        is_classifier=False,
        transformers=model["transformers"],
    )
    su_dte = SurrogateDecisionTree(
        model=model["model"],
        transformers=model["transformers"],
    )

    dte.fit(x)
    su_dte.fit(x)

    helper_produce_decision_tree_regression_one_hot(dte, model)
    helper_produce_decision_tree_regression_one_hot(su_dte, model)


def helper_produce_decision_tree_regression_one_hot(explainer, model):
    tree_object = explainer.produce().get()
    assert tree_object.feature_importances_.shape == (
        explainer.transform_to_x_algorithm(model["x"]).shape[1],
    )


def test_produce_decision_tree_classification_no_transforms(classification_no_transform_tree):
    model = classification_no_transform_tree
    dte = DecisionTreeExplainer(
        model=model["model"],
        x_train_orig=model["x"],
        e_algorithm="surrogate_tree",
        is_classifier=True,
        transformers=model["transformers"],
        fit_on_init=True,
        classes=np.arange(2),
    )
    SUdte = SurrogateDecisionTree(
        model=model["model"],
        x_train_orig=model["x"],
        transformers=model["transformers"],
        fit_on_init=True,
        classes=np.arange(2),
    )

    helper_produce_decision_tree_classification_no_transforms(
        dte, classification_no_transform_tree
    )
    helper_produce_decision_tree_classification_no_transforms(
        SUdte, classification_no_transform_tree
    )


def helper_produce_decision_tree_classification_no_transforms(explainer, model):
    tree_object = explainer.produce().get()
    assert tree_object.feature_importances_.shape == (
        explainer.transform_to_x_algorithm(model["x"]).shape[1],
    )
    assert (tree_object.predict(model["x"].to_numpy()) == model["y"].to_numpy().ravel()).all()


def test_produce_decision_tree_regression_no_transforms_with_size(regression_no_transforms):
    model = regression_no_transforms
    dte = DecisionTreeExplainer(
        model=model["model"],
        x_train_orig=model["x"],
        e_algorithm="surrogate_tree",
        is_classifier=False,
        max_depth=5,
        transformers=model["transformers"],
        fit_on_init=True,
        training_size=2,
    )
    SUdte = SurrogateDecisionTree(
        model=model["model"],
        x_train_orig=model["x"],
        is_classifier=False,
        max_depth=5,
        transformers=model["transformers"],
        fit_on_init=True,
        training_size=2,
    )

    helper_produce_decision_tree_regression_no_transforms(dte, model)
    helper_produce_decision_tree_regression_no_transforms(SUdte, model)


def test_produce_decision_tree_regression_transforms_with_size(regression_one_hot):
    model = regression_one_hot
    dte = DecisionTreeExplainer(
        model=model["model"],
        x_train_orig=model["x"],
        e_algorithm="surrogate_tree",
        is_classifier=False,
        transformers=model["transformers"],
        fit_on_init=True,
        training_size=2,
    )
    SUdte = SurrogateDecisionTree(
        model=model["model"],
        x_train_orig=model["x"],
        transformers=model["transformers"],
        fit_on_init=True,
        training_size=2,
    )

    helper_produce_decision_tree_regression_one_hot(dte, model)
    helper_produce_decision_tree_regression_one_hot(SUdte, model)


def test_produce_decision_tree_classification_with_size(classification_no_transform_tree):
    model = classification_no_transform_tree
    dte = DecisionTreeExplainer(
        model=model["model"],
        x_train_orig=model["x"],
        e_algorithm="surrogate_tree",
        is_classifier=True,
        transformers=model["transformers"],
        fit_on_init=True,
        training_size=2,
        classes=np.arange(2),
    )
    SUdte = SurrogateDecisionTree(
        model=model["model"],
        x_train_orig=model["x"],
        transformers=model["transformers"],
        fit_on_init=True,
        training_size=2,
        classes=np.arange(2),
    )

    helper_produce_decision_tree_classification_no_transforms_with_size(
        dte, classification_no_transform_tree
    )
    helper_produce_decision_tree_classification_no_transforms_with_size(
        SUdte, classification_no_transform_tree
    )


def helper_produce_decision_tree_classification_no_transforms_with_size(explainer, model):
    tree_object = explainer.produce().get()
    assert tree_object.feature_importances_.shape == (
        explainer.transform_to_x_algorithm(model["x"]).shape[1],
    )
