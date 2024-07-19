import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from pyreal import RealApp
from pyreal.transformers import NarrativeTransformer


def test_prepare_local_feature_contribution(regression_no_transforms):
    real_app = RealApp(
        regression_no_transforms["model"],
        transformers=regression_no_transforms["transformers"],
    )
    x = pd.DataFrame([[2, 10, 10]])

    with pytest.raises(ValueError):
        real_app.produce_feature_contributions(x)

    # Confirm no error
    real_app.prepare_feature_contributions(
        x_train_orig=regression_no_transforms["x"], y_train=regression_no_transforms["y"]
    )

    # Confirm explainer was prepped and now works without being given data
    real_app.produce_feature_contributions(x)


def test_prepare_local_feature_contribution_with_id_column(regression_no_transforms):
    real_app = RealApp(
        regression_no_transforms["model"],
        transformers=regression_no_transforms["transformers"],
        id_column="ID",
    )
    features = ["A", "B", "C"]
    x_multi_dim = pd.DataFrame([[4, 1, 1, "a"], [6, 2, 3, "b"]], columns=features + ["ID"])

    # Confirm no error
    real_app.prepare_feature_contributions(
        x_train_orig=x_multi_dim, y_train=regression_no_transforms["y"]
    )

    # Confirm explainer was prepped and now works without being given data
    real_app.produce_feature_contributions(x_multi_dim)


def test_produce_local_feature_contributions(regression_no_transforms):
    real_app = RealApp(
        regression_no_transforms["model"],
        regression_no_transforms["x"],
        transformers=regression_no_transforms["transformers"],
    )
    features = ["A", "B", "C"]

    x_one_dim = pd.Series([2, 10, 10], index=features)

    expected = np.mean(regression_no_transforms["y"])
    explanation = real_app.produce_feature_contributions(x_one_dim)

    assert list(explanation["Feature Name"]) == features
    assert list(explanation["Feature Value"]) == list(x_one_dim)
    assert list(explanation["Contribution"]) == [x_one_dim.iloc[0] - expected, 0, 0]

    x_multi_dim = pd.DataFrame([[2, 1, 1], [4, 2, 3]], columns=features)
    explanation = real_app.produce_feature_contributions(x_multi_dim)

    assert list(explanation[0]["Feature Name"]) == features
    assert list(explanation[0]["Feature Value"]) == list(x_multi_dim.iloc[0])
    assert list(explanation[0]["Contribution"]) == [x_multi_dim.iloc[0, 0] - expected, 0, 0]

    assert list(explanation[1]["Feature Name"]) == features
    assert list(explanation[1]["Feature Value"]) == list(x_multi_dim.iloc[1])
    assert list(explanation[1]["Contribution"]) == [x_multi_dim.iloc[1, 0] - expected, 0, 0]


def test_produce_local_feature_contributions_with_averages(regression_no_transforms):
    real_app = RealApp(
        regression_no_transforms["model"],
        regression_no_transforms["x"],
        transformers=regression_no_transforms["transformers"],
    )
    features = ["A", "B", "C"]

    x_multi_dim = pd.DataFrame([[2, 1, 1], [4, 2, 3]], columns=features)
    explanation = real_app.produce_feature_contributions(x_multi_dim, include_average_values=True)
    averages = list(regression_no_transforms["x"].mean(axis="rows"))

    assert list(explanation[0]["Average/Mode"]) == averages
    assert list(explanation[1]["Average/Mode"]) == averages


def test_produce_local_feature_contributions_no_format(
    regression_no_transforms, regression_one_hot
):
    real_app = RealApp(
        regression_no_transforms["model"],
        regression_no_transforms["x"],
        transformers=regression_no_transforms["transformers"],
        id_column="ID",
    )
    features = ["A", "B", "C"]

    x_multi_dim = pd.DataFrame([[2, 1, 1, "a"], [4, 2, 3, "b"]], columns=features + ["ID"])

    expected = np.mean(regression_no_transforms["y"])
    explanation, values = real_app.produce_feature_contributions(x_multi_dim, format_output=False)
    assert explanation.shape == x_multi_dim.drop(columns="ID").shape
    assert list(x_multi_dim["ID"]) == list(explanation.index)

    assert_series_equal(
        explanation["A"], x_multi_dim["A"] - expected, check_index=False, check_index_type=False
    )
    assert (explanation["B"] == 0).all()
    assert (explanation["C"] == 0).all()

    assert_frame_equal(values, x_multi_dim.set_index("ID"))
    assert list(x_multi_dim["ID"]) == list(values.index)

    x_multi_dim = pd.DataFrame([[2, 1, 1], [4, 2, 3]], columns=features)
    real_app = RealApp(
        regression_one_hot["model"],
        regression_one_hot["x"],
        transformers=regression_one_hot["transformers"],
    )

    _, values = real_app.produce_feature_contributions(x_multi_dim, format_output=False)
    assert_frame_equal(values, x_multi_dim)


def test_produce_local_feature_contributions_with_id_column(regression_one_hot):
    real_app = RealApp(
        regression_one_hot["model"],
        regression_one_hot["x"],
        transformers=regression_one_hot["transformers"],
        id_column="ID",
    )

    features = ["A", "B", "C"]
    x_one_dim = pd.Series([4, 1, 1, "ab"], index=features + ["ID"])
    explanation = real_app.produce_feature_contributions(x_one_dim)
    explanation_a1 = explanation.sort_values(by="Feature Name", axis=0)

    x_multi_dim = pd.DataFrame([[4, 1, 1, "a"], [6, 2, 3, "b"]], columns=features + ["ID"])

    explanation = real_app.produce_feature_contributions(x_multi_dim)

    explanation_a2 = explanation["a"].sort_values(by="Feature Name", axis=0)
    explanation_b = explanation["b"].sort_values(by="Feature Name", axis=0)

    for explanation_a in [explanation_a1, explanation_a2]:
        assert list(explanation_a["Feature Name"]) == features
        assert list(explanation_a["Feature Value"]) == list(x_multi_dim.iloc[0])[:-1]
        for num in list(explanation_a["Contribution"]):
            assert abs(num) < 0.001

    assert list(explanation_b["Feature Name"]) == features
    assert list(explanation_b["Feature Value"]) == list(x_multi_dim.iloc[1])[:-1]
    assert abs(list(explanation_b["Contribution"])[0] - 1) < 0.001
    for num in list(explanation_a["Contribution"][1:]):
        assert abs(num) < 0.001


def test_produce_local_feature_contributions_with_index_names(regression_one_hot):
    real_app = RealApp(
        regression_one_hot["model"],
        regression_one_hot["x"],
        transformers=regression_one_hot["transformers"],
    )

    features = ["A", "B", "C"]
    x_one_dim = pd.Series([4, 1, 1], index=features, name="ab")
    explanation = real_app.produce_feature_contributions(x_one_dim)
    explanation_a1 = explanation.sort_values(by="Feature Name", axis=0)

    x_multi_dim = pd.DataFrame([[4, 1, 1], [6, 2, 3]], columns=features, index=["a", "b"])

    explanation = real_app.produce_feature_contributions(x_multi_dim)

    explanation_a2 = explanation["a"].sort_values(by="Feature Name", axis=0)
    explanation_b = explanation["b"].sort_values(by="Feature Name", axis=0)

    for explanation_a in [explanation_a1, explanation_a2]:
        assert list(explanation_a["Feature Name"]) == features
        assert list(explanation_a["Feature Value"]) == list(x_multi_dim.iloc[0])
        for num in list(explanation_a["Contribution"]):
            assert abs(num) < 0.001

    assert list(explanation_b["Feature Name"]) == features
    assert list(explanation_b["Feature Value"]) == list(x_multi_dim.iloc[1])
    assert abs(list(explanation_b["Contribution"])[0] - 1) < 0.001
    for num in list(explanation_a["Contribution"][1:]):
        assert abs(num) < 0.001


def test_produce_local_feature_contributions_no_data_on_init(regression_no_transforms):
    real_app = RealApp(
        regression_no_transforms["model"],
        transformers=regression_no_transforms["transformers"],
    )
    features = ["A", "B", "C"]
    x_one_dim = pd.DataFrame([[2, 10, 10]], columns=features)

    expected = np.mean(regression_no_transforms["y"])
    explanation = real_app.produce_feature_contributions(
        x_one_dim, x_train_orig=regression_no_transforms["x"]
    )

    assert list(explanation[0]["Feature Name"]) == features
    assert list(explanation[0]["Feature Value"]) == list(x_one_dim.iloc[0])
    assert list(explanation[0]["Contribution"]) == [x_one_dim.iloc[0, 0] - expected, 0, 0]


def test_produce_local_feature_contributions_num_features(regression_no_transforms):
    real_app = RealApp(
        regression_no_transforms["model"],
        regression_no_transforms["x"],
        transformers=regression_no_transforms["transformers"],
    )

    x_one_dim = pd.DataFrame([[2, 10, 10]])
    explanation = real_app.produce_feature_contributions(
        x_one_dim, num_features=2, select_by="min"
    )
    assert explanation[0].shape == (2, 3)


def test_produce_narrative_feature_contributions(regression_one_hot, mock_openai_client):
    real_app = RealApp(
        regression_one_hot["model"],
        regression_one_hot["x"],
        transformers=regression_one_hot["transformers"],
        openai_client=mock_openai_client["client"],
    )
    features = ["A", "B", "C"]
    x_one_dim = pd.DataFrame([[2, 10, 10]], columns=features)

    explanation = real_app.produce_narrative_feature_contributions(x_one_dim)
    assert explanation[0] == mock_openai_client["response"]

    x_multi_dim = pd.DataFrame([[2, 1, 1], [4, 2, 3]], columns=features, index=["a", "b"])
    explanation = real_app.produce_narrative_feature_contributions(x_multi_dim)
    assert explanation["a"] == mock_openai_client["response"]
    assert explanation["b"] == mock_openai_client["response"]

    x_series = pd.Series([2, 10, 10], index=features)
    explanation = real_app.produce_narrative_feature_contributions(x_series)
    assert explanation[0] == mock_openai_client["response"]


def test_produce_narrative_feature_contributions_with_id_column(
    regression_one_hot, mock_openai_client
):
    real_app = RealApp(
        regression_one_hot["model"],
        regression_one_hot["x"],
        transformers=regression_one_hot["transformers"],
        openai_client=mock_openai_client["client"],
        id_column="ID",
    )
    features = ["A", "B", "C", "ID"]
    x_one_dim = pd.DataFrame([[2, 10, 10, "A"]], columns=features)

    explanation = real_app.produce_narrative_feature_contributions(x_one_dim)
    assert explanation["A"] == mock_openai_client["response"]

    x_multi_dim = pd.DataFrame(
        [[2, 1, 1, "A"], [4, 2, 3, "B"]], columns=features, index=["a", "b"]
    )
    explanation = real_app.produce_narrative_feature_contributions(x_multi_dim)
    assert explanation["A"] == mock_openai_client["response"]
    assert explanation["B"] == mock_openai_client["response"]

    x_series = pd.Series([2, 10, 10, "A"], index=features)
    explanation = real_app.produce_narrative_feature_contributions(x_series)
    assert explanation[0] == mock_openai_client["response"]


def test_produce_narrative_feature_contributions_optimized(regression_one_hot, mock_openai_client):
    real_app = RealApp(
        regression_one_hot["model"],
        regression_one_hot["x"],
        transformers=regression_one_hot["transformers"],
        openai_client=mock_openai_client["client"],
        id_column="ID",
    )
    features = ["A", "B", "C", "ID"]
    x_multi_dim = pd.DataFrame(
        [[2, 1, 1, "A"], [4, 2, 3, "B"]], columns=features, index=["a", "b"]
    )

    explanation = real_app.produce_narrative_feature_contributions(
        x_multi_dim, format_output=False
    )

    assert explanation[0] == mock_openai_client["response"]
    assert explanation[1] == mock_openai_client["response"]


def test_train_llm(regression_one_hot, mock_openai_client, mocker):
    real_app = RealApp(
        regression_one_hot["model"],
        regression_one_hot["x"],
        transformers=regression_one_hot["transformers"],
        openai_client=mock_openai_client["client"],
        id_column="ID",
    )

    def custom_input(prompt):
        if "Save training data? (y/n)" in prompt:
            return "y"
        return "example explanation"

    mocker.patch("builtins.input", side_effect=custom_input)
    mocker.patch("builtins.print")  # disable printing for cleaner logs
    real_app.train_feature_contribution_llm(num_inputs=2, provide_examples=True)
    for algorithm in real_app.explainers["lfc"]:
        narratives = real_app.explainers["lfc"][algorithm].llm_training_data
        mocker.stopall()
        assert len(narratives) == 2
        assert narratives[0][1] == "example explanation"
        assert narratives[1][1] == "example explanation"


def test_train_llm_input_transformer(regression_one_hot, mock_openai_client, mocker):
    real_app = RealApp(
        regression_one_hot["model"],
        regression_one_hot["x"],
        transformers=regression_one_hot["transformers"],
        openai_client=mock_openai_client["client"],
        id_column="ID",
    )
    input_transformer = NarrativeTransformer(
        openai_client=mock_openai_client["client"], num_features=3
    )

    def custom_input(prompt):
        if "Save training data? (y/n)" in prompt:
            return "y"
        return "example explanation"

    mocker.patch("builtins.input", side_effect=custom_input)
    mocker.patch("builtins.print")  # disable printing for cleaner logs
    real_app.train_feature_contribution_llm(
        num_inputs=2, provide_examples=True, transformer=input_transformer
    )
    narratives = input_transformer.training_examples["feature_contributions"]
    mocker.stopall()
    assert len(narratives) == 2
    assert narratives[0][1] == "example explanation"
    assert narratives[1][1] == "example explanation"
