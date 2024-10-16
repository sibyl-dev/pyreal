import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal

from pyreal.explainers import LocalFeatureContribution


def test_produce_with_renames(regression_one_hot):
    model = regression_one_hot
    transforms = model["transformers"]
    feature_descriptions = {"A": "Feature A", "B": "Feature B"}
    lfc = LocalFeatureContribution(
        model=model["model"],
        x_train_orig=model["x"],
        e_algorithm="shap",
        fit_on_init=True,
        transformers=transforms,
        feature_descriptions=feature_descriptions,
    )
    x_one_dim = pd.DataFrame([[2, 10, 10]], columns=["A", "B", "C"])

    explanation = lfc.produce(x_one_dim)
    contributions = explanation.get()
    assert x_one_dim.shape == contributions.shape
    assert abs(contributions["Feature A"][0] + 1) < 0.0001
    assert abs(contributions["Feature B"][0]) < 0.0001
    assert abs(contributions["C"][0]) < 0.0001
    means_with_descriptions = model["x"].mean().rename(feature_descriptions)
    assert_series_equal(explanation.get_average_values(), means_with_descriptions)


def test_evaluate_variation(classification_no_transforms):
    model = classification_no_transforms
    lfc = LocalFeatureContribution(
        model=model["model"],
        x_train_orig=model["x"],
        e_algorithm="shap",
        transformers=model["transformers"],
        fit_on_init=True,
        classes=np.arange(1, 4),
    )

    # Assert no crash. Values analyzed through benchmarking
    lfc.evaluate_variation(with_fit=False, n_iterations=5)
    lfc.evaluate_variation(with_fit=True, n_iterations=5)


def test_produce_with_renames_with_size(regression_one_hot):
    model = regression_one_hot
    transforms = model["transformers"]
    feature_descriptions = {"A": "Feature A", "B": "Feature B"}
    lfc = LocalFeatureContribution(
        model=model["model"],
        x_train_orig=model["x"],
        e_algorithm="shap",
        fit_on_init=True,
        transformers=transforms,
        feature_descriptions=feature_descriptions,
        training_size=2,
    )
    x_one_dim = pd.DataFrame([[2, 10, 10]], columns=["A", "B", "C"])

    contributions = lfc.produce(x_one_dim).get()
    assert x_one_dim.shape == contributions.shape


def test_evaluate_variation_with_size(classification_no_transforms):
    model = classification_no_transforms
    lfc = LocalFeatureContribution(
        model=model["model"],
        x_train_orig=model["x"],
        e_algorithm="shap",
        transformers=model["transformers"],
        fit_on_init=True,
        training_size=4,
        classes=np.arange(1, 4),
    )

    # Assert no crash. Values analyzed through benchmarking
    lfc.evaluate_variation(with_fit=False, n_iterations=5)
    lfc.evaluate_variation(with_fit=True, n_iterations=5)


def test_produce_narrative_explanation(regression_one_hot, mock_narrator):
    lfc = LocalFeatureContribution(
        model=regression_one_hot["model"],
        x_train_orig=regression_one_hot["x"],
        e_algorithm="shap",
        fit_on_init=True,
        transformers=regression_one_hot["transformers"],
        narrator=mock_narrator["narrator"],
    )

    x_one_dim = pd.DataFrame([[2, 10, 10]], columns=["A", "B", "C"])
    explanation = lfc.produce_narrative_explanation(x_one_dim).get()
    assert explanation[0] == mock_narrator["response"]

    x_multi_dim = pd.DataFrame([[2, 10, 10], [2, 11, 11]], columns=["A", "B", "C"])
    explanation = lfc.produce_narrative_explanation(x_multi_dim).get()
    assert explanation[0] == mock_narrator["response"]
    assert explanation[1] == mock_narrator["response"]


def test_produce_narrative_explanation_with_training(regression_one_hot, mock_narrator):
    lfc = LocalFeatureContribution(
        model=regression_one_hot["model"],
        x_train_orig=regression_one_hot["x"],
        e_algorithm="shap",
        fit_on_init=True,
        transformers=regression_one_hot["transformers"],
        narrator=mock_narrator["narrator"],
    )

    training_data = [("(f1, 1, 1), (f2, 2, 2", "the model uses f1 and f2")]
    lfc.set_llm_training_data(training_data)
    assert lfc.llm_training_data is training_data

    x_one_dim = pd.DataFrame([[2, 10, 10]], columns=["A", "B", "C"])
    explanation = lfc.produce_narrative_explanation(x_one_dim).get()
    assert explanation[0] == mock_narrator["response"]

    lfc.clear_llm_training_data()
    assert lfc.llm_training_data is None


def test_train_llm(regression_one_hot, mock_narrator, mocker):
    lfc = LocalFeatureContribution(
        model=regression_one_hot["model"],
        x_train_orig=regression_one_hot["x"],
        e_algorithm="shap",
        fit_on_init=True,
        transformers=regression_one_hot["transformers"],
        narrator=mock_narrator["narrator"],
    )

    def custom_input(prompt):
        if "Save training data? (y/n)" in prompt:
            return "y"
        return "example explanation"

    mocker.patch("builtins.input", side_effect=custom_input)
    mocker.patch("builtins.print")  # disable printing for cleaner logs
    narratives = lfc.train_llm(num_inputs=2, provide_examples=True)
    mocker.stopall()
    assert len(narratives) == 2
    assert narratives[0][1] == "example explanation"
    assert narratives[1][1] == "example explanation"
    assert lfc.llm_training_data == narratives
