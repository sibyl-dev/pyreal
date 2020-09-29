from shap import KernelExplainer, LinearExplainer
from sibyl.explainers.base import Explainer
import numpy as np
import pickle
from abc import ABC, abstractmethod
import pandas as pd


class LocalFeatureContribution(Explainer):
    def __init__(self, e_algorithm="shap", contribution_transformers=None, **kwargs):
        """
        Initial a FeatureContributions object
        :param e_algorithm: string
               Explanation algorithm to use
        :param contribution_transformers: contribution transformer object(s)
               Object or list of objects that include .transform_contributions(contributions)
               functions, used to adjust the contributions back to interpretable form.
        :param model_pickle_filepath: filepath
               Filepath to the pickled model to explain
        :param X_orig: dataframe of shape (n_instances, x_orig_feature_count)
               The training set for the explainer
        :param y_orig: dataframe of shape (n_instances,)
               The y values for the dataset
        :param feature_descriptions: dict
               Interpretable descriptions of each feature
        :param e_transforms: transformer object or list of transformer objects
               Transformer(s) that need to be used on x_orig for the explanation algorithm:
                    x_orig -> x_explain
        :param m_transforms: transformer object or list of transformer objects
               Transformer(s) needed on x_orig to make predictions on the dataset with model, if different
               than ex_transforms
                    x_orig -> x_model
        :param i_transforms: transformer object or list of transformer objects
               Transformer(s) needed to make x_orig interpretable
                    x_orig -> x_interpret
        :param combination_transformer: transformer object
               Transformer used to combine
        :param fit_on_init: Boolean
               If True, fit the explainer on initiation.
               If False, self.fit() must be manually called before produce() is called
        """
        super(LocalFeatureContribution, self).__init__(**kwargs)

        # TODO: add some functionality to automatically pick e_algorithm
        if e_algorithm is None:
            e_algorithm = choose_algorithm(self.model)
        if e_algorithm not in ["shap"]:
            raise ValueError("Algorithm %s not understood" % e_algorithm)

        if not isinstance(contribution_transformers, list):
            self.contribution_transformers = [contribution_transformers]
        else:
            self.contribution_transformers = contribution_transformers

        self.algorithm = e_algorithm
        self.explainer = None

    def fit(self):
        """
        Fit the contribution explainer
        """
        # TODO: if model is linear sklearn, set explainer type to linear
        explainer_type = "linear"
        if self.algorithm == "shap":
            dataset = self.transform_to_x_explain(self.X_orig)
            if explainer_type is "kernel":
                self.explainer = KernelExplainer(self.model.predict, dataset)
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
        if x.ndim == 1:
            x = x.to_frame().T
        print(x.shape)
        if self.explainer is None:
            raise AttributeError("Instance has no explainer. Must call "
                                 "fit_contribution_explainer before "
                                 "get_contributions")
        if x.shape[1] != self.expected_feature_number:
            raise ValueError("Received input of wrong size."
                             "Expected ({},), received {}"
                             .format(self.expected_feature_number, x.shape))
        if self.algorithm == "shap":
            x = self.transform_to_x_explain(x)
            columns = x.columns
            x = np.asanyarray(x)
            contributions = pd.DataFrame(self.explainer.shap_values(x), columns=columns)
            return self.transform_contributions(contributions)

    def transform_contributions(self, contributions):
        """
        Transform contributions to interpretable form
        :param contributions: DataFrame of shape (n_instances, x_explain_feature_count)
        :return: DataFrame of shape (n_instances, x_interpret_feature_count)
        """
        if self.contribution_transformers is None:
            return contributions
        for transform in self.contribution_transformers:
            contributions = transform.transform_contributions(contributions)
        return contributions


def choose_algorithm(model):
    """
    Choose an algorithm based on the model type.
    Current, shap is the only supported algorithm
    :param model: model object
    :return: one of ["shap"]
    """
    return "shap"




