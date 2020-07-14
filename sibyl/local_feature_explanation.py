import shap
import numpy as np
import pickle


def fit_contribution_explainer(model, X_train,
                               savefile=None, return_result=False):
    """
    Fit a shap explainer.

    :param model: sklearn.linear_model
          The model to explain
    :param X_train: array_like of shape (n_samples, n_features)
          The training set for the model
    :param savefile: file object
          Where the save the explainer. If None, don't save
    :param return_result: boolean
          If true, return the resulting explainer, else return none
    :return: explainer or None
             Returns the explainer if return_result is True
    """
    # TODO: Update this to be model agnostic
    X_train = np.asanyarray(X_train)
    explainer = shap.LinearExplainer(model, X_train)
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


def get_contributions(x, explainer):
    """
    Get the feature contributions for all features in x.

    :param x: array_like of shape (n_features,)
           The input into the model
    :param explainer: pretrained SHAP explainer
    :return: array_like of shape (n_features,)
             The contributions of each feature in x
    """
    x = np.asanyarray(x)
    shap_values = explainer.shap_values(x)
    return shap_values


