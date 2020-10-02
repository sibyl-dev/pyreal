from shap import Explainer as ShapExplainer, LinearExplainer, KernelExplainer
from sibyl.explainers.base import Explainer
import numpy as np
import pandas as pd


class LocalFeatureContribution(Explainer):
    """
    LocalFeatureContributions explainer object

    A LocalFeatureContributions object explains a machine learning prediction by assigning an
    importance or contribution score to every feature. LocalFeatureContribution objects explain by
    taking an instance and returning one number per feature, per instance.

    Args:
        model_pickle_filepath (filepath string):
           Filepath to the pickled model to explain
        X_orig (dataframe of shape (n_instances, x_orig_feature_count)):
           The training set for the explainer
        y_orig (dataframe of shape (n_instances,)):
           The y values for the dataset
        feature_descriptions (dict):
           Interpretable descriptions of each feature
        e_transforms (transformer object or list of transformer objects):
           Transformer(s) that need to be used on x_orig for the explanation algorithm:
                x_orig -> x_explain
        m_transforms (transformer object or list of transformer objects):
           Transformer(s) needed on x_orig to make predictions on the dataset with model, if different
           than ex_transforms
                x_orig -> x_model
        i_transforms (transformer object or list of transformer objects):
           Transformer(s) needed to make x_orig interpretable
                x_orig -> x_interpret
        fit_on_init (Boolean):
           If True, fit the explainer on initiation.
           If False, self.fit() must be manually called before produce() is called
        e_algorithm (string, one of ["shap"]):
           Explanation algorithm to use. If none, one will be chosen automatically based on model
           type
        contribution_transformers (contribution transformer object(s)):
           Object or list of objects that include .transform_contributions(contributions)
           functions, used to adjust the contributions back to interpretable form.
    """
    def __init__(self, model_pickle_filepath, X_orig,
                 e_algorithm="shap", contribution_transformers=None, **kwargs):
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
        Calculate the contributions of each feature in x

        Args:
            x (DataFrame of shape (n_instances, n_features)):
               The input to be explained
        Returns:
            DataFrame of shape (n_instances, n_features):
                 The contribution of each feature
        """
        return self.base_local_feature_contribution.produce(x)

    def transform_contributions(self, contributions):
        """
        Transform contributions to an interpretable form.

        Args:
            contributions (DataFrame of shape (n_instances, x_explain_feature_count)):
        Returns:
            DataFrame of shape (n_instances, x_interpret_feature_count)
                The transformed contributions
        """
        if self.contribution_transformers is None:
            return contributions
        for transform in self.contribution_transformers:
            contributions = transform.transform_contributions(contributions)
        return contributions


class ShapFeatureContribution(Explainer):
    """
    ShapFeatureContribution object.

    A ShapFeatureContribution object gets feature contributions using the SHAP algorithm.

    Args:
        model_pickle_filepath (string filepath):
            Filepath to the pickled model to explain
        X_orig (DataFrame of size (n_instances, n_features)):
            Training set in original form.
        contribution_transformers (transformer or list of transformers):
            Transformer that convert contributions from explanation form to interpretable form
        shap_type (string, one of ["kernel", "linear"]):
            Type of shap algorithm to use. If None, SHAP will pick one.
        **kwargs: see base Explainer args
    """
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
        Calculate the contributions of each feature in x using SHAP.

        Args:
            x (DataFrame of shape (n_instances, n_features)):
               The input to be explained
        Returns:
            DataFrame of shape (n_instances, n_features):
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
        Transform contributions to an interpretable form.

        Args:
            contributions (DataFrame of shape (n_instances, x_explain_feature_count)):
        Returns:
            DataFrame of shape (n_instances, x_interpret_feature_count)
                The transformed contributions
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

    Args:
        model (model object)
            Model to be explained
    Return:
        string (one of ["shap"])
            Explanation algorithm to use
    """
    return "shap"




