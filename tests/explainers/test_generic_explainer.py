import numpy as np

from pyreal.explainers import Explainer


def test_predict_regression_global_shap(regression_no_transforms, regression_one_hot):
    model = regression_no_transforms
    explainer = Explainer(model["model"], model["x"],
                          e_algorithm="shap",
                          m_transformers=model["transformers"])
    expected = np.array(model["y"]).reshape(-1)
    result = explainer.model_predict(model["x"])
    assert np.array_equal(result, expected)

    model = regression_one_hot
    explainer = Explainer(model["model"], model["x"],
                          e_algorithm="shap",
                          m_transformers=model["transformers"])
    expected = np.array(model["y"]).reshape(-1)
    result = explainer.model_predict(model["x"])
    assert np.array_equal(result, expected)


def test_predict_classification_global_shap(classification_no_transforms):
    model = classification_no_transforms
    explainer = Explainer(model["model"], model["x"],
                          e_algorithm="shap",
                          m_transformers=model["transformers"])
    expected = np.array(model["y"])
    result = explainer.model_predict(model["x"])
    assert np.array_equal(result, expected)


def test_evaluate_model_global_shap(regression_no_transforms):
    explainer = Explainer(regression_no_transforms["model"],
                          regression_no_transforms["x"],
                          e_algorithm="shap",
                          y_orig=regression_no_transforms["y"])
    score = explainer.evaluate_model("accuracy")
    assert score == 1

    score = explainer.evaluate_model("neg_mean_squared_error")
    assert score == 0

    new_y = regression_no_transforms["x"].iloc[:, 0:1].copy()
    new_y.iloc[0, 0] = 0
    explainer = Explainer(regression_no_transforms["model"],
                          regression_no_transforms["x"],
                          e_algorithm="shap",
                          y_orig=new_y)
    score = explainer.evaluate_model("accuracy")
    assert abs(score - .6667) <= 0.0001


def test_predict_regression_local_shap(regression_no_transforms, regression_one_hot):
    model = regression_no_transforms
    explainer = Explainer(model["model"], model["x"],
                          scope="local",
                          e_algorithm="shap",
                          m_transformers=model["transformers"])
    expected = np.array(model["y"]).reshape(-1)
    result = explainer.model_predict(model["x"])
    assert np.array_equal(result, expected)

    model = regression_one_hot
    explainer = Explainer(model["model"], model["x"],
                          scope="local",
                          e_algorithm="shap",
                          m_transformers=model["transformers"])
    expected = np.array(model["y"]).reshape(-1)
    result = explainer.model_predict(model["x"])
    assert np.array_equal(result, expected)


def test_predict_classification_local_shap(classification_no_transforms):
    model = classification_no_transforms
    explainer = Explainer(model["model"], model["x"],
                          scope="local",
                          e_algorithm="shap",
                          m_transformers=model["transformers"])
    expected = np.array(model["y"])
    result = explainer.model_predict(model["x"])
    assert np.array_equal(result, expected)


def test_evaluate_model_local_shap(regression_no_transforms):
    explainer = Explainer(regression_no_transforms["model"],
                          regression_no_transforms["x"],
                          scope="local",
                          e_algorithm="shap",
                          y_orig=regression_no_transforms["y"])
    score = explainer.evaluate_model("accuracy")
    assert score == 1

    score = explainer.evaluate_model("neg_mean_squared_error")
    assert score == 0

    new_y = regression_no_transforms["x"].iloc[:, 0:1].copy()
    new_y.iloc[0, 0] = 0
    explainer = Explainer(regression_no_transforms["model"],
                          regression_no_transforms["x"],
                          e_algorithm="shap",
                          y_orig=new_y)
    score = explainer.evaluate_model("accuracy")
    assert abs(score - .6667) <= 0.0001
