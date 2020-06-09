import shap
import lime.lime_tabular
import numpy as np


class LocalFeatureContributions:

    def __init__(self):
        self.explainer = None

    def fit_contributions(self, model, X_train):
        """
        Fit the shap explainer.

        :param model: sklearn.linear_model
               The model to explain
        :param X_train: array_like of shape (n_samples, n_features)
               The training set for the model
        :return: None
        """
        # TODO: Update this to be model agnostic
        X_train = np.asanyarray(X_train)
        self.explainer = shap.LinearExplainer(model, X_train)

    def get_contributions(self, x):
        """
        Get the feature contributions for all features in x.
        Must call fit_shap before calling this function.

        :param x: array_like of shape (n_features,)
               The input into the model
        :return: array_like of shape (n_features,)
                 The contributions of each feature in x
        :except: AssertError
                 If fit_contributions has not been called
        """
        assert self.explainer is not None, \
            "Need to call fit_contributions before calling get_contributions"
        x = np.asanyarray(x)
        shap_values = self.explainer.shap_values(x)
        return shap_values


class LimeExplanation:

    def __init__(self):
        self.explainer = None

    def fit_contributions(self, X_train, feature_names=None):
        X_train = np.asanyarray(X_train)
        self.explainer = lime.lime_tabular.LimeTabularExplainer(X_train,
                                                            mode="regression",
                                                            feature_names=feature_names)

    def get_contributions(self, x, predict, num_features=5):
        assert self.explainer is not None, \
            "Need to call fit_contributions before calling get_contributions"
        x = np.asanyarray(x)
        explanation = self.explainer.explain_instance(x, predict,
                                                      num_features=num_features)
        print(explanation.as_list())


