from shap import KernelExplainer, LinearExplainer
from sibyl.explainers.base import Explainer
import numpy as np
import pickle
from abc import ABC, abstractmethod


class LocalFeatureContribution(Explainer):
    def __init__(self, *args, e_algorithm="shap"):
        """
        Initial a FeatureContributions object
        :param model: model object
               The model to explain
        :param x_orig: dataframe of shape (n_instances, n_features)
               The training set for the explainer
        :param y_orig: dataframe of shape (n_instances,)
               The y values for the dataset
        :param e_transforms: transformer object or list of transformer objects
               Transformer(s) needed to run explanation algorithm on dataset
        :param m_transforms: transformer object of list of transformer objects
               Transformer(s) needed to make predictions on the dataset with model, if different
               than e_transforms
        :param e_algorithm: one of ["shap"]
        :param fit_on_init: Boolean
               If True, fit the feature contribution explainer on initiation.
               If False, explainer will be set to None and must be fit before
                         get_contributions is called
        """
        super(LocalFeatureContribution, self).__init__(*args)

        # TODO: add some functionality to automatically pick e_algorithm
        if e_algorithm is None:
            e_algorithm = choose_algorithm(self.model)
        if e_algorithm not in ["shap"]:
            raise ValueError("Algorithm %s not understood" % e_algorithm)

        self.algorithm = e_algorithm
        self.explainer = None

    def fit(self):
        """
        Fit the contribution explainer
        """
        # TODO: if model is linear sklearn, set explainer type to linear
        explainer_type = "kernel"
        if self.algorithm == "shap":
            dataset = self.transform_to_x_explain(np.asanyarray(self.X_orig))
            if explainer_type is "kernel":
                self.explainer = KernelExplainer(self.model, dataset)
            elif explainer_type is "linear":
                self.explainer = LinearExplainer(self.model, dataset)
            else:
                raise ValueError("Unrecognized shap type %s" % type)

    def produce(self, x):
        """
        Returns the contributions of each feature in x
        :param x: DataFrame of shape (n_instances, n_features)
               The input to be explained
        :return: DataFrame of shape (n_instances, n_features
                 The contribution of each feature
        """
        if self.explainer is None:
            raise AttributeError("Instance has no explainer. Must call "
                                 "fit_contribution_explainer before "
                                 "get_contributions")
        if x.shape[1] != self.expected_feature_number:
            raise ValueError("Received input of wrong size."
                             "Expected ({},), received {}"
                             .format(self.expected_feature_number, x.shape))
        if self.algorithm == "shap":
            x = np.asanyarray(x)
            contributions = self.explainer.shap_values(x)
            return contributions


def choose_algorithm(model):
    """
    Choose an algorithm based on the model type.
    Current, shap is the only supported algorithm
    :param model: model object
    :return: one of ["shap"]
    """
    return "shap"




