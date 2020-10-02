from shap import Explainer as ShapExplainer, LinearExplainer, KernelExplainer
from sibyl.explainers.base import Explainer
import numpy as np
import pickle
from abc import ABC, abstractmethod
import pandas as pd


class LocalFeatureContribution(Explainer):
    def __init__(self, model_pickle_filepath, X_orig,
                 e_algorithm="shap", contribution_transformers=None, **kwargs):
        """
        Initial a FeatureContributions object
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
        :param fit_on_init: Boolean
               If True, fit the explainer on initiation.
               If False, self.fit() must be manually called before produce() is called
        :param e_algorithm: string
               Explanation algorithm to use
        :param contribution_transformers: contribution transformer object(s)
               Object or list of objects that include .transform_contributions(contributions)
               functions, used to adjust the contributions back to interpretable form.
        """
        if e_algorithm is None:
            e_algorithm = choose_algorithm(self.model)
        if e_algorithm not in ["shap"]:
            raise ValueError("Algorithm %s not understood" % e_algorithm)

        if contribution_transformers is not None and \
                not isinstance(contribution_transformers, list):
            self.contribution_transformers = [contribution_transformers]
        else:
            self.contribution_transformers = contribution_transformers

        if e_algorithm is "shap":
            self.base_local_feature_contribution = ShapFeatureContribution(
                model_pickle_filepath, X_orig,
                contribution_transformers, **kwargs)

        super(LocalFeatureContribution, self).__init__(model_pickle_filepath, X_orig, **kwargs)

    def fit(self):
        """
        Fit the contribution explainer
        """
        self.base_local_feature_contribution.fit()

    def produce(self, x):
        """
        Returns the contributions of each feature in x
        :param x: DataFrame of shape (n_instances, n_features)
               The input to be explained
        :return: DataFrame of shape (n_instances, n_features
                 The contribution of each feature
        """
        return self.base_local_feature_contribution.produce(x)

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


class ShapFeatureContribution(Explainer):
    def __init__(self, model_pickle_filepath, X_orig,
                 contribution_transformers=None, shap_type=None, **kwargs):
        if contribution_transformers is not None and \
                not isinstance(contribution_transformers, list):
            self.contribution_transformers = [contribution_transformers]
        else:
            self.contribution_transformers = contribution_transformers
        supported_types = ["kernel", "linear"]
        if shap_type is not None and shap_type not in supported_types:
            raise ValueError("Shap type not supported, given %s, expected one of %s or None" %
                  (shap_type, str(supported_types)))
        else:
            self.shap_type = shap_type

        self.explainer = None
        super(ShapFeatureContribution, self).__init__(model_pickle_filepath, X_orig, **kwargs)

    def fit(self):
        """
        Fit the contribution explainer
        """
        dataset = self.transform_to_x_explain(self.X_orig)
        if self.shap_type == "kernel":
            self.explainer = KernelExplainer(self.model.predict, dataset)
        # Note: we manually check for linear model here because of SHAP bug
        elif self.shap_type is "linear" or LinearExplainer.supports_model(self.model):
            self.explainer = LinearExplainer(self.model, dataset)
        else:
            self.explainer = ShapExplainer(self.model, dataset)  # SHAP will pick an algorithm

    def produce(self, x):
        """
        Returns the contributions of each feature in x
        :param x: DataFrame of shape (n_instances, n_features)
               The input to be explained
        :return: DataFrame of shape (n_instances, n_features
                 The contribution of each feature
        """
        if x.ndim == 1:
            x = x.to_frame().reshape(1, -1)
        if self.explainer is None:
            raise AttributeError("Instance has no explainer. Must call "
                                 "fit_contribution_explainer before "
                                 "get_contributions")
        if x.shape[1] != self.expected_feature_number:
            raise ValueError("Received input of wrong size."
                             "Expected ({},), received {}"
                             .format(self.expected_feature_number, x.shape))
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
    Currently, shap is the only supported algorithm
    :param model: model object
    :return: one of ["shap"]
    """
    return "shap"




