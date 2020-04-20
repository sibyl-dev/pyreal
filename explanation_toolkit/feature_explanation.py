import shap

explainer = None


def fit_contributions(model, X_train):
    """
    Fit the shap explainer.

    :param model: the model to explain
    :param X_train: the standardized training set to calculate the contributions
    :return: None
    """
    global explainer
    explainer = shap.LinearExplainer(model, X_train)


def get_contributions(x):
    """
    Get the feature contributions for all features in x.
    Must call fit_shap before calling this function.

    :param x: a standardized input into the model
    :return: the contributions of each feature in x
    :except: AssertError if fit_contributions has not been called
    """
    assert(explainer is not None, "Need to call fit_contributions before " +
                                  "calling get_contributions")
    shap_values = explainer.shap_values(x)
    return shap_values
