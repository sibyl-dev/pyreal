import numpy as np
import pandas as pd
from shap import LinearExplainer

from pyreal.explainers import (
    LocalFeatureContribution, ShapFeatureContribution, SimpleCounterfactualContribution,)


def test_fit_shap(all_models):
    for model in all_models:
        lfc_object = LocalFeatureContribution(
            model=model["model"],
            x_train_orig=model["x"], transformers=model["transformers"],
            e_algorithm='shap')
        lfc_object.fit()
        shap = ShapFeatureContribution(
            model=model["model"],
            x_train_orig=model["x"], transformers=model["transformers"])
        shap.fit()

        assert shap.explainer is not None
        assert isinstance(shap.explainer, LinearExplainer)


def test_fit_simple(all_models):
    for model in all_models:
        simple = SimpleCounterfactualContribution(
            model=model["model"],
            x_train_orig=model["x"], transformers=model["transformers"])
        # Assert no error
        simple.fit()


def test_produce_shap_regression_no_transforms(regression_no_transforms):
    model = regression_no_transforms
    lfc = LocalFeatureContribution(model=model["model"],
                                   x_train_orig=model["x"], e_algorithm='shap',
                                   transformers=model["transformers"],
                                   fit_on_init=True)
    shap = ShapFeatureContribution(
        model=model["model"], x_train_orig=model["x"], transformers=model["transformers"],
        fit_on_init=True)

    helper_produce_shap_regression_no_transforms(lfc, model)
    helper_produce_shap_regression_no_transforms(shap, model)


def helper_produce_shap_regression_no_transforms(explainer, model):
    x_one_dim = pd.DataFrame([[2, 10, 10]], columns=["A", "B", "C"])
    x_multi_dim = pd.DataFrame([[2, 1, 1],
                                [4, 2, 3]], columns=["A", "B", "C"])
    expected = np.mean(model["y"])[0]
    contributions = explainer.produce(x_one_dim)[0]
    assert x_one_dim.shape == contributions.shape
    assert contributions.iloc[0, 0] == x_one_dim.iloc[0, 0] - expected
    assert contributions.iloc[0, 1] == 0
    assert contributions.iloc[0, 2] == 0

    contributions = explainer.produce(x_multi_dim)[0]
    assert x_multi_dim.shape == contributions.shape
    assert contributions.iloc[0, 0] == x_multi_dim.iloc[0, 0] - expected
    assert contributions.iloc[1, 0] == x_multi_dim.iloc[1, 0] - expected
    assert (contributions.iloc[:, 1] == 0).all()
    assert (contributions.iloc[:, 2] == 0).all()


def test_produce_simple_regression_no_transforms(regression_no_transforms):
    model = regression_no_transforms
    explainer = SimpleCounterfactualContribution(model=model["model"],
                                                 x_train_orig=model["x"],
                                                 transformers=model["transformers"],
                                                 fit_on_init=True)
    x_one_dim = pd.DataFrame([[2, 10, 10]], columns=["A", "B", "C"])
    x_multi_dim = pd.DataFrame([[2, 1, 1],
                                [4, 2, 3]], columns=["A", "B", "C"])
    contributions = explainer.produce(x_one_dim)[0]
    assert x_one_dim.shape == contributions.shape
    assert contributions.iloc[0, 0] <= 4
    assert contributions.iloc[0, 0] >= .01  # with very high probability
    assert contributions.iloc[0, 1] == 0
    assert contributions.iloc[0, 2] == 0

    contributions = explainer.produce(x_multi_dim)[0]
    assert x_multi_dim.shape == contributions.shape
    assert contributions.iloc[0, 0] <= 4
    assert contributions.iloc[0, 0] >= .01  # with very high probability
    assert contributions.iloc[1, 0] <= 2
    assert contributions.iloc[1, 0] > .5
    assert (contributions.iloc[:, 1] == 0).all()
    assert (contributions.iloc[:, 2] == 0).all()


def test_produce_shap_regression_transforms(regression_one_hot):
    model = regression_one_hot
    lfc = LocalFeatureContribution(model=model["model"],
                                   x_train_orig=model["x"], e_algorithm='shap',
                                   transformers=model["transformers"],
                                   fit_on_init=True)
    shap = ShapFeatureContribution(
        model=model["model"], x_train_orig=model["x"], transformers=model["transformers"],
        fit_on_init=True)

    helper_produce_shap_regression_one_hot(lfc)
    helper_produce_shap_regression_one_hot(shap)


def helper_produce_shap_regression_one_hot(explainer):
    x_one_dim = pd.DataFrame([[2, 10, 10]], columns=["A", "B", "C"])
    x_multi_dim = pd.DataFrame([[4, 1, 1],
                                [6, 2, 3]], columns=["A", "B", "C"])
    contributions = explainer.produce(x_one_dim)[0]
    assert x_one_dim.shape == contributions.shape
    assert abs(contributions["A"][0] + 1) < .0001
    assert abs(contributions["B"][0]) < .0001
    assert abs(contributions["C"][0]) < .0001

    contributions = explainer.produce(x_multi_dim)[0]
    assert x_multi_dim.shape == contributions.shape
    assert abs(contributions["A"][0]) < .0001
    assert abs(contributions["A"][1] - 1 < .0001)
    assert (contributions["B"] == 0).all()
    assert (contributions["C"] == 0).all()


def test_produce_simple_regression_transforms(regression_one_hot):
    model = regression_one_hot
    model["transformers"].set_flags(algorithm=False)
    explainer = SimpleCounterfactualContribution(model=model["model"],
                                                 x_train_orig=model["x"],
                                                 transformers=model["transformers"],
                                                 fit_on_init=True)
    x_one_dim = pd.DataFrame([[2, 10, 10]], columns=["A", "B", "C"])
    x_multi_dim = pd.DataFrame([[4, 1, 1],
                                [6, 2, 3]], columns=["A", "B", "C"])
    contributions = explainer.produce(x_one_dim)[0]
    assert x_one_dim.shape == contributions.shape
    assert contributions["A"][0] <= 2
    assert contributions["A"][0] >= .5
    assert contributions["B"][0] == 0
    assert contributions["C"][0] == 0

    contributions = explainer.produce(x_multi_dim)[0]
    print(contributions)
    assert x_multi_dim.shape == contributions.shape
    assert contributions["A"][0] <= 2
    assert contributions["A"][0] >= .01  # with high probability
    assert contributions["A"][1] <= 2
    assert contributions["A"][1] >= .01  # with high probability
    assert (contributions["B"] == 0).all()
    assert (contributions["C"] == 0).all()


def test_produce_shap_classification_no_transforms(classification_no_transforms):
    model = classification_no_transforms
    lfc = LocalFeatureContribution(model=model["model"],
                                   x_train_orig=model["x"], e_algorithm='shap',
                                   transformers=model["transformers"],
                                   fit_on_init=True,
                                   classes=np.arange(1, 4))
    shap = ShapFeatureContribution(
        model=model["model"], x_train_orig=model["x"], transformers=model["transformers"],
        fit_on_init=True, classes=np.arange(1, 4))

    helper_produce_shap_classification_no_transforms(lfc)
    helper_produce_shap_classification_no_transforms(shap)


def helper_produce_shap_classification_no_transforms(explainer):
    x_one_dim = pd.DataFrame([[1, 0, 0]], columns=["A", "B", "C"])
    x_multi_dim = pd.DataFrame([[1, 0, 0],
                                [1, 1, 0]], columns=["A", "B", "C"])
    contributions = explainer.produce(x_one_dim)[0]
    assert x_one_dim.shape == contributions.shape
    assert abs(contributions["A"][0]) < .0001
    assert abs(contributions["B"][0] + 1) < .0001
    assert abs(contributions["C"][0]) < .0001

    contributions = explainer.produce(x_multi_dim)[0]
    assert x_multi_dim.shape == contributions.shape
    assert (contributions["A"] == 0).all()
    assert abs(contributions["B"][0] + 1) < .0001
    assert abs(contributions["B"][1]) < .0001
    assert (contributions["C"] == 0).all()


def test_produce_with_renames(regression_one_hot):
    model = regression_one_hot
    transforms = model["transformers"]
    feature_descriptions = {"A": "Feature A", "B": "Feature B"}
    lfc = LocalFeatureContribution(model=model["model"],
                                   x_train_orig=model["x"], e_algorithm='shap',
                                   fit_on_init=True, transformers=transforms,
                                   interpretable_features=True,
                                   feature_descriptions=feature_descriptions)
    x_one_dim = pd.DataFrame([[2, 10, 10]], columns=["A", "B", "C"])

    contributions = lfc.produce(x_one_dim)[0]
    assert x_one_dim.shape == contributions.shape
    assert abs(contributions["Feature A"][0] + 1) < 0.0001
    assert abs(contributions["Feature B"][0]) < 0.0001
    assert abs(contributions["C"][0]) < 0.0001


def test_evaluate_variation(classification_no_transforms):
    model = classification_no_transforms
    lfc = LocalFeatureContribution(model=model["model"],
                                   x_train_orig=model["x"], e_algorithm='shap',
                                   transformers=model["transformers"],
                                   fit_on_init=True,
                                   classes=np.arange(1, 4))

    # Assert no crash. Values analyzed through benchmarking
    lfc.evaluate_variation(with_fit=False, n_iterations=5)
    lfc.evaluate_variation(with_fit=True, n_iterations=5)


def test_fit_shap_with_size(all_models):
    for model in all_models:
        shap_with_size = ShapFeatureContribution(
            model=model["model"],
            x_train_orig=model["x"], transformers=model["transformers"], training_size=2)
        shap_with_size.fit()

        assert shap_with_size.explainer is not None
        assert isinstance(shap_with_size.explainer, LinearExplainer)


def test_fit_simple_with_size(all_models):
    for model in all_models:
        simple = SimpleCounterfactualContribution(
            model=model["model"],
            x_train_orig=model["x"], transformers=model["transformers"], training_size=2)
        # Assert no error
        simple.fit()


def test_produce_shap_regression_no_transforms_with_size(regression_no_transforms):
    model = regression_no_transforms

    shap = ShapFeatureContribution(
        model=model["model"], x_train_orig=model["x"], transformers=model["transformers"],
        fit_on_init=True, training_size=2)

    helper_produce_shap_regression_no_transforms_with_size(shap, model)


def helper_produce_shap_regression_no_transforms_with_size(explainer, model):
    x_one_dim = pd.DataFrame([[2, 10, 10]], columns=["A", "B", "C"])
    x_multi_dim = pd.DataFrame([[2, 1, 1],
                                [4, 2, 3]], columns=["A", "B", "C"])

    contributions = explainer.produce(x_one_dim)[0]
    assert x_one_dim.shape == contributions.shape
    contributions = explainer.produce(x_multi_dim)[0]
    assert x_multi_dim.shape == contributions.shape
    assert (contributions.iloc[:, 1] == 0).all()
    assert (contributions.iloc[:, 2] == 0).all()


def test_produce_simple_regression_no_transforms_with_size(regression_no_transforms):
    model = regression_no_transforms
    explainer = SimpleCounterfactualContribution(model=model["model"],
                                                 x_train_orig=model["x"],
                                                 transformers=model["transformers"],
                                                 fit_on_init=True,
                                                 training_size=1)
    x_one_dim = pd.DataFrame([[2, 10, 10]], columns=["A", "B", "C"])
    x_multi_dim = pd.DataFrame([[2, 1, 1],
                                [4, 2, 3]], columns=["A", "B", "C"])
    contributions = explainer.produce(x_one_dim)[0]
    assert x_one_dim.shape == contributions.shape
    assert contributions.iloc[0, 1] == 0
    assert contributions.iloc[0, 2] == 0

    contributions = explainer.produce(x_multi_dim)[0]
    assert x_multi_dim.shape == contributions.shape
    assert (contributions.iloc[:, 1] == 0).all()
    assert (contributions.iloc[:, 2] == 0).all()


def test_produce_shap_regression_transforms_with_size(regression_one_hot):
    model = regression_one_hot

    shap = ShapFeatureContribution(
        model=model["model"], x_train_orig=model["x"], transformers=model["transformers"],
        fit_on_init=True, training_size=2)

    helper_produce_shap_regression_one_hot_with_size(shap)


def helper_produce_shap_regression_one_hot_with_size(explainer):
    x_one_dim = pd.DataFrame([[2, 10, 10]], columns=["A", "B", "C"])
    x_multi_dim = pd.DataFrame([[4, 1, 1],
                                [6, 2, 3]], columns=["A", "B", "C"])
    contributions = explainer.produce(x_one_dim)[0]
    assert x_one_dim.shape == contributions.shape
    assert abs(contributions["B"][0]) < .0001
    assert abs(contributions["C"][0]) < .0001

    contributions = explainer.produce(x_multi_dim)[0]
    assert x_multi_dim.shape == contributions.shape
    assert (contributions["B"] == 0).all()
    assert (contributions["C"] == 0).all()


def test_produce_shap_classification_no_transforms_with_size(classification_no_transforms):
    model = classification_no_transforms
    shap = ShapFeatureContribution(
        model=model["model"], x_train_orig=model["x"], transformers=model["transformers"],
        fit_on_init=True, classes=np.arange(1, 4), training_size=3)

    helper_produce_shap_classification_no_transforms_with_size(shap)


def helper_produce_shap_classification_no_transforms_with_size(explainer):
    x_one_dim = pd.DataFrame([[1, 0, 0]], columns=["A", "B", "C"])
    x_multi_dim = pd.DataFrame([[1, 0, 0],
                                [1, 0, 0]], columns=["A", "B", "C"])
    contributions = explainer.produce(x_one_dim)[0]
    assert x_one_dim.shape == contributions.shape

    contributions = explainer.produce(x_multi_dim)[0]
    print(contributions)
    assert x_multi_dim.shape == contributions.shape


def test_produce_with_renames_with_size(regression_one_hot):
    model = regression_one_hot
    transforms = model["transformers"]
    feature_descriptions = {"A": "Feature A", "B": "Feature B"}
    lfc = LocalFeatureContribution(model=model["model"],
                                   x_train_orig=model["x"], e_algorithm='shap',
                                   fit_on_init=True, transformers=transforms,
                                   interpretable_features=True,
                                   feature_descriptions=feature_descriptions,
                                   training_size=2)
    x_one_dim = pd.DataFrame([[2, 10, 10]], columns=["A", "B", "C"])

    contributions = lfc.produce(x_one_dim)[0]
    assert x_one_dim.shape == contributions.shape


def test_evaluate_variation_with_size(classification_no_transforms):
    model = classification_no_transforms
    lfc = LocalFeatureContribution(model=model["model"],
                                   x_train_orig=model["x"], e_algorithm='shap',
                                   transformers=model["transformers"],
                                   fit_on_init=True,
                                   training_size=4,
                                   classes=np.arange(1, 4))

    # Assert no crash. Values analyzed through benchmarking
    lfc.evaluate_variation(with_fit=False, n_iterations=5)
    lfc.evaluate_variation(with_fit=True, n_iterations=5)
