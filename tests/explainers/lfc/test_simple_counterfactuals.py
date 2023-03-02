import pandas as pd

from pyreal.explainers import SimpleCounterfactualContribution


def test_fit_simple(all_models):
    for model in all_models:
        simple = SimpleCounterfactualContribution(
            model=model["model"], x_train_orig=model["x"], transformers=model["transformers"]
        )
        # Assert no error
        simple.fit()


def test_fit_simple_with_size(all_models):
    for model in all_models:
        simple = SimpleCounterfactualContribution(
            model=model["model"],
            x_train_orig=model["x"],
            transformers=model["transformers"],
            training_size=2,
        )
        # Assert no error
        simple.fit()


def test_produce_simple_regression_no_transforms(regression_no_transforms):
    model = regression_no_transforms
    explainer = SimpleCounterfactualContribution(
        model=model["model"],
        x_train_orig=model["x"],
        transformers=model["transformers"],
        fit_on_init=True,
    )
    x_one_dim = pd.DataFrame([[2, 10, 10]], columns=["A", "B", "C"])
    x_multi_dim = pd.DataFrame([[2, 1, 1], [4, 2, 3]], columns=["A", "B", "C"])
    contributions = explainer.produce(x_one_dim)[0]
    assert x_one_dim.shape == contributions.shape
    assert contributions.iloc[0, 0] <= 4
    assert contributions.iloc[0, 0] >= 0.01  # with very high probability
    assert contributions.iloc[0, 1] == 0
    assert contributions.iloc[0, 2] == 0

    contributions = explainer.produce(x_multi_dim)[0]
    assert x_multi_dim.shape == contributions.shape
    assert contributions.iloc[0, 0] <= 4
    assert contributions.iloc[0, 0] >= 0.01  # with very high probability
    assert contributions.iloc[1, 0] <= 2
    assert contributions.iloc[1, 0] > 0.5
    assert (contributions.iloc[:, 1] == 0).all()
    assert (contributions.iloc[:, 2] == 0).all()


def test_produce_simple_regression_transforms(regression_one_hot):
    model = regression_one_hot
    model["transformers"].set_flags(algorithm=False)
    explainer = SimpleCounterfactualContribution(
        model=model["model"],
        x_train_orig=model["x"],
        transformers=model["transformers"],
        fit_on_init=True,
    )
    x_one_dim = pd.DataFrame([[2, 10, 10]], columns=["A", "B", "C"])
    x_multi_dim = pd.DataFrame([[4, 1, 1], [6, 2, 3]], columns=["A", "B", "C"])
    contributions = explainer.produce(x_one_dim)[0]
    assert x_one_dim.shape == contributions.shape
    assert contributions["A"][0] <= 2
    assert contributions["A"][0] >= 0.5
    assert contributions["B"][0] == 0
    assert contributions["C"][0] == 0

    contributions = explainer.produce(x_multi_dim)[0]
    print(contributions)
    assert x_multi_dim.shape == contributions.shape
    assert contributions["A"][0] <= 2
    assert contributions["A"][0] >= 0.01  # with high probability
    assert contributions["A"][1] <= 2
    assert contributions["A"][1] >= 0.01  # with high probability
    assert (contributions["B"] == 0).all()
    assert (contributions["C"] == 0).all()


def test_produce_simple_regression_no_transforms_with_size(regression_no_transforms):
    model = regression_no_transforms
    explainer = SimpleCounterfactualContribution(
        model=model["model"],
        x_train_orig=model["x"],
        transformers=model["transformers"],
        fit_on_init=True,
        training_size=1,
    )
    x_one_dim = pd.DataFrame([[2, 10, 10]], columns=["A", "B", "C"])
    x_multi_dim = pd.DataFrame([[2, 1, 1], [4, 2, 3]], columns=["A", "B", "C"])
    contributions = explainer.produce(x_one_dim)[0]
    assert x_one_dim.shape == contributions.shape
    assert contributions.iloc[0, 1] == 0
    assert contributions.iloc[0, 2] == 0

    contributions = explainer.produce(x_multi_dim)[0]
    assert x_multi_dim.shape == contributions.shape
    assert (contributions.iloc[:, 1] == 0).all()
    assert (contributions.iloc[:, 2] == 0).all()
