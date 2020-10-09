from shap import Explainer as ShapExplainer, LinearExplainer, KernelExplainer
from real.explainers.base import Explainer
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
import pickle


class LocalFeatureContributionsBase(Explainer, ABC):
    """
    Base class for LocalFeatureContributionsBase explainer objects

    A LocalFeatureContributionsBase object explains a machine learning prediction by assigning an
    importance or contribution score to every feature. LocalFeatureContribution objects explain by
    taking an instance and returning one number per feature, per instance.

    Args:
        model_pickle_filepath (filepath string):
           Filepath to the pickled model to explain
        x_orig (dataframe of shape (n_instances, x_orig_feature_count)):
           The training set for the explainer
        e_algorithm (string, one of ["shap"]):
           Explanation algorithm to use. If none, one will be chosen automatically based on model
           type
        contribution_transformers (contribution transformer object(s)):
           Object or list of objects that include .transform_contributions(contributions)
           functions, used to adjust the contributions back to interpretable form.
        interpretable_features (Boolean):
            If True, return explanations using the interpretable feature descriptions instead of
            default names
        **kwargs: see base Explainer args
    """
    def __init__(self, model_pickle_filepath, x_orig,
                 contribution_transformers=None, interpretable_features=True, **kwargs):
        if contribution_transformers is not None and \
                not isinstance(contribution_transformers, list):
            self.contribution_transformers = [contribution_transformers]
        else:
            self.contribution_transformers = contribution_transformers

        self.interpretable_features = interpretable_features
        super(LocalFeatureContributionsBase, self).__init__(model_pickle_filepath, x_orig,
                                                            **kwargs)

    @abstractmethod
    def fit(self):
        """
        Fit this explainer object
        """
        pass

    def produce(self, x_orig):
        """
        Produce the local feature contribution explanation

        Args:
            x_orig (DataFrame of shape (n_instances, n_features):
                Input to explain

        Returns:
            DataFrame of shape (n_instances, n_features)
                Contribution of each feature for each instance
        """
        if x_orig.ndim == 1:
            x_orig = x_orig.to_frame().reshape(1, -1)
        if x_orig.shape[1] != self.expected_feature_number:
            raise ValueError("Received input of wrong size."
                             "Expected ({},), received {}"
                             .format(self.expected_feature_number, x_orig.shape))
        contributions = self.get_contributions(x_orig)
        return self.transform_contributions(contributions)

    @abstractmethod
    def get_contributions(self, x_orig):
        """
        Gets the raw contributions. Abstract method.
        Args:
            x_orig (DataFrame of shape (n_instances, n_features):
                Input to explain

        Returns:
            DataFrame of shape (n_instances, n_features)
                Contribution of each feature for each instance
        """
        pass

    def transform_contributions(self, contributions):
        """
        Transform contributions to an interpretable form.

        Args:
            contributions (DataFrame of shape (n_instances, x_explain_feature_count)):
            interpretable_features (Boolean):
                If True, convert column names to interpretable description form before returning
        Returns:
            DataFrame of shape (n_instances, x_interpret_feature_count)
                The transformed contributions
        """
        if self.contribution_transformers is None:
            return contributions
        for transform in self.contribution_transformers:
            contributions = transform.transform_contributions(contributions)
        if self.interpretable_features:
            return self.convert_columns_to_interpretable(contributions)
        return contributions


class LocalFeatureContribution(LocalFeatureContributionsBase):
    """
    Generic LocalFeatureContribution wrapper

    A LocalFeatureContributions object wraps multiple local feature-based explanations. If no
    specific algorithm is requested, one will be chosen based on the information given.
    Currently, only SHAP is supported.

    Args:
        model_pickle_filepath (filepath string):
           Filepath to the pickled model to explain
        X_orig (dataframe of shape (n_instances, x_orig_feature_count)):
           The training set for the explainer
        e_algorithm (string, one of ["shap"]):
           Explanation algorithm to use. If none, one will be chosen automatically based on model
           type
        contribution_transformers (contribution transformer object(s)):
           Object or list of objects that include .transform_contributions(contributions)
           functions, used to adjust the contributions back to interpretable form.
        **kwargs: see base Explainer args
    """

    def __init__(self, model_pickle_filepath, x_orig,
                 contribution_transformers=None, e_algorithm=None, **kwargs):
        if e_algorithm is None:
            e_algorithm = choose_algorithm()

        self.base_local_feature_contribution = None
        if e_algorithm is "shap":
            self.base_local_feature_contribution = ShapFeatureContribution(
                model_pickle_filepath, x_orig,
                contribution_transformers=contribution_transformers, **kwargs)
        if self.base_local_feature_contribution is None:
            raise ValueError("Invalid algorithm type %s" % e_algorithm)

        super(LocalFeatureContribution, self).__init__(model_pickle_filepath, x_orig,
                                                       contribution_transformers, **kwargs)

    def fit(self):
        """
        Fit this explainer object
        """
        self.base_local_feature_contribution.fit()

    def get_contributions(self, x_orig):
        """
        Gets the raw contributions.
        Args:
            x_orig (DataFrame of shape (n_instances, n_features):
                Input to explain

        Returns:
            DataFrame of shape (n_instances, n_features)
                Contribution of each feature for each instance
        """
        return self.base_local_feature_contribution.get_contributions(x_orig)


class ShapFeatureContribution(LocalFeatureContributionsBase):
    """
    ShapFeatureContribution object.

    A ShapFeatureContribution object gets feature contributions using the SHAP algorithm.

    Args:
        model_pickle_filepath (string filepath):
            Filepath to the pickled model to explain
        x_orig (DataFrame of size (n_instances, n_features)):
            Training set in original form.
        contribution_transformers (transformer or list of transformers):
            Transformer that convert contributions from explanation form to interpretable form
        shap_type (string, one of ["kernel", "linear"]):
            Type of shap algorithm to use. If None, SHAP will pick one.
        **kwargs: see base Explainer args
    """
    def __init__(self, model_pickle_filepath, x_orig,
                 shap_type=None, **kwargs):
        supported_types = ["kernel", "linear"]
        if shap_type is not None and shap_type not in supported_types:
            raise ValueError("Shap type not supported, given %s, expected one of %s or None" %
                  (shap_type, str(supported_types)))
        else:
            self.shap_type = shap_type

        self.explainer = None
        super(ShapFeatureContribution, self).__init__(model_pickle_filepath, x_orig, **kwargs)

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

    def get_contributions(self, x_orig):
        """
        Calculate the contributions of each feature in x using SHAP.

        Args:
            x_orig (DataFrame of shape (n_instances, n_features)):
               The input to be explained
        Returns:
            DataFrame of shape (n_instances, n_features):
                 The contribution of each feature
        """
        if self.explainer is None:
            raise AttributeError("Instance has no explainer. Must call "
                                 "fit_contribution_explainer before "
                                 "get_contributions")
        x = self.transform_to_x_explain(x_orig)
        columns = x.columns
        x = np.asanyarray(x)
        return pd.DataFrame(self.explainer.shap_values(x), columns=columns)

# ------------------------------------------------------ #
#  HELPERS                                               #
# ------------------------------------------------------ #

def choose_algorithm():
    """
    Choose an algorithm based on the model type.
    Currently, shap is the only supported algorithm

    Return:
        string (one of ["shap"])
            Explanation algorithm to use
    """
    return "shap"

# ------------------------------------------------------ #
#  FUNCTIONAL IMPLEMENTATIONS                            #
# ------------------------------------------------------ #


def fit_and_produce_local_feature_contributions(model_pickle_filepath, x_input, x_train,
                                                contribution_transformers=None,
                                                e_algorithm=None, feature_descriptions=None,
                                                e_transforms=None, m_transforms=None,
                                                i_transforms=None, interpretable_features=True):
    """
    Get a local feature contribution for x_input

    Args:
        model_pickle_filepath (filepath string):
           Filepath to the pickled model to explain
        x_input (dataframe of shape (n_instances, x_orig_feature_count)):
           The input to explain
        x_train (dataframe of shape (n_instances, x_orig_feature_count)):
           The training set for the explainer
        contribution_transformers (contribution transformer object(s)):
           Object or list of objects that include .transform_contributions(contributions)
           functions, used to adjust the contributions back to interpretable form.
        e_algorithm (string, one of ["shap"]):
           Explanation algorithm to use. If none, one will be chosen automatically based on model
           type
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
        interpretable_features (Boolean):
            If True, return explanations using the interpretable feature descriptions instead of
            default names

    Returns:
        DataFrame of shape (n_instances, n_features):
            The contribution of each feature
    """
    lfc = LocalFeatureContribution(model_pickle_filepath, x_train,
                                   contribution_transformers=contribution_transformers,
                                   e_algorithm=e_algorithm,
                                   feature_descriptions=feature_descriptions,
                                   e_transforms=e_transforms, m_transforms=m_transforms,
                                   i_transforms=i_transforms,
                                   interpretable_features=interpretable_features,
                                   fit_on_init=True)
    return lfc.produce(x_input)


def fit_local_feature_contributions(model_pickle_filepath, x_train,
                                    save_filepath=None, contribution_transformers=None,
                                    e_algorithm=None, feature_descriptions=None,
                                    e_transforms=None, m_transforms=None,
                                    i_transforms=None, interpretable_features=True):
    """
    Fit a local feature contributions object that can later be used to get explanations

    Args:
        model_pickle_filepath (filepath string):
           Filepath to the pickled model to explain
        x_train (dataframe of shape (n_instances, x_orig_feature_count)):
           The training set for the explainer
        save_filepath (string):
            Location to save the pickled object return. If none, do not sav
        contribution_transformers (contribution transformer object(s)):
           Object or list of objects that include .transform_contributions(contributions)
           functions, used to adjust the contributions back to interpretable form.
        e_algorithm (string, one of ["shap"]):
           Explanation algorithm to use. If none, one will be chosen automatically based on model
           type
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
        interpretable_features (Boolean):
            If True, return explanations using the interpretable feature descriptions instead of
            default names

    Returns:
        LocalFeatureContribution object
            Fitted local feature contribution object
    """
    # TODO: should we return the full object, or just the explainer?
    lfc = LocalFeatureContribution(model_pickle_filepath, x_train,
                                   contribution_transformers=contribution_transformers,
                                   e_algorithm=e_algorithm,
                                   feature_descriptions=feature_descriptions,
                                   e_transforms=e_transforms, m_transforms=m_transforms,
                                   i_transforms=i_transforms,
                                   interpretable_features=interpretable_features,
                                   fit_on_init=True)
    if save_filepath is not None:
        with open(save_filepath, "wb") as f:
            pickle.dump(lfc, f)
    return lfc


def produce_feature_contributions(x_orig, local_feature_contribution):
    """
    Get a local feature contribution from a fitted explainer

    Args:
        x_orig (DataFrame of shape (n_instances, n_features):
            Input to explain
        local_feature_contribution (LocalFeatureContribution object):
            Fitted explainer object

    Returns:
        DataFrame of shape (n_instances, n_features):
            The contribution of each feature
    """
    return local_feature_contribution.produce(x_orig)

