import shap
import numpy as np
import pandas as pd
import pickle


def fit_contribution_explainer(model, X_train, transformer=None,
                               savefile=None, return_result=False,
                               use_linear_explainer=False):
    """
    Fit a shap explainer.

    :param model: sklearn.linear_model (if "use_linear_explainer" is true) or predictor with .predict function
          The model to explain
    :param X_train: array_like of shape (n_samples, n_features)
          The training set for the model
    :param transformer: transformer object
           Transformer object to use before training explainer
    :param savefile: file object
          Where the save the explainer. If None, don't save
    :param return_result: boolean
          If true, return the resulting explainer, else return none
    :param use_linear_explainer: boolean
          If true, use a linear explainer (faster, but required sklean linear model input)
    :return: explainer or None
             Returns the explainer if return_result is True
    """
    if transformer is not None:
        X_train = transformer.transform(X_train)
    if use_linear_explainer:
        explainer = shap.LinearExplainer(model, X_train)
    else:
        explainer = shap.KernelExplainer(model.predict, X_train)
    if savefile is not None:
        pickle.dump(explainer, savefile)
    if return_result:
        return explainer


def load_contribution_explainer(file):
    """
    Load a contribution explainer.

    :param file: file object
           The file of the pickled explainer
    :return: explainer object
             The explainer
    """
    return pickle.load(file)


def get_contributions(X, explainer, transformer=None):
    """
    Get the feature contributions for all features in x.

    :param X: DataFrame of shape (n_features,)
           The input into the model
    :param transformer: transformer object
           Transformer object to use before training explainer
    :param explainer: pretrained SHAP explainer
    :return: array_like of shape (n_features,)
             The contributions of each feature in X
    """
    if transformer is not None:
        X = transformer.transform(X)
    columns = X.columns
    shap_values = explainer.shap_values(X)
    shap_values = pd.DataFrame(shap_values, columns=columns)
    if transformer is not None:
        shap_values = transformer.transform_contributions_shap(shap_values)
    return shap_values


