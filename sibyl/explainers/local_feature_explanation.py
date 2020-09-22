from shap import KernelExplainer, LinearExplainer
import numpy as np
import pickle

# CLASS WRAPPERS ----------------


class FeatureContributionExplainer:
    def __init__(self, model, dataset, transformers=None,
                 algorithm="shap", fit_on_init=True):
        """
        Initial a FeatureContributions object
        :param model: model object
               The model to explain
        :param dataset: dataframe of shape (n_instances, n_features)
               The training set for the explainer
        :param transformers: transformer object or list of transformer objects
               Transformer(s) to use before getting contributions
        :param algorithm: one of ["shap"]
        :param fit_on_init: Boolean
               If True, fit the feature contribution explainer on initiation.
               If False, explainer will be set to None and must be fit before
                         get_contributions is called
        """
        self.base_explainer = None
        # TODO: check is model has .predict function
        # TODO: check if transformer(s) have transform
        # TODO: add some functionality to automatically pick algorithm
        if not isinstance(transformers, list):
            self.transformers = [transformers]
        else:
            self.transformers = transformers
        self.expected_feature_number = dataset.shape[1]

        if algorithm is None:
            algorithm = choose_algorithm(model)
        if algorithm not in ["shap"]:
            raise ValueError("Algorithm %s not understood" % algorithm)

        self.algorithm = algorithm
        self.model = model
        self.dataset = dataset

        if self.transformers is not None:
            for transformer in self.transformers:
                dataset = transformer.transform(dataset)

        self.explainer = None

        if fit_on_init:
            self.fit_contributions()

    def fit_contributions(self):
        if self.algorithm == "shap":
            # TODO: if model is linear sklearn, set explainer type to linear
            self.explainer = fit_contributions_shap(self.model, self.dataset,
                                                    savefile=None, return_result=True,
                                                    explainer_type="linear")

    def get_contributions(self, x):
        """
        Returns the contributions of each feature in x
        :param x:
        :return:
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
            return get_contributions_shap(x, self.explainer)


# STATELESS IMPLEMENTATIONS --------------

def choose_algorithm(model):
    """
    Choose an algorithm based on the model type.
    Current, shap is the only supported algorithm
    :param model: model object
    :return: one of ["shap"]
    """
    return "shap"


def fit_contributions_shap(model, dataset,
                           savefile=None, return_result=False,
                           explainer_type="kernel"):
    """
    Fit a contribution explainer for the shap algorithm.

    :param model: function array-like (dataset.shape) -> (dataset.shape[0])
          The prediction function of the model to explain
    :param dataset: array-like of shape (n_instances, n_features)
          The training set for the model
    :param savefile: file object
          Where the save the explainer. If None, don't save
    :param return_result: boolean
          If true, return the resulting explainer, else return none
    :param explainer_type: one of ["kernel", "linear"]
           The type of shap explainer to fit
    :return: explainer or None
             Returns the explainer if return_result is True
    """
    dataset = np.asanyarray(dataset)
    if explainer_type is "kernel":
        explainer = KernelExplainer(model, dataset)
    elif explainer_type is "linear":
        explainer = LinearExplainer(model, dataset)
    else:
        raise ValueError("Unrecognized shap type %s" % type)
    if savefile is not None:
        pickle.dump(explainer, savefile)
    if return_result:
        return explainer


def get_contributions_shap(x, explainer):
    """
    Get the feature contributions for all features in x.

    :param x: array_like of shape (n_features,)
           The input into the model
    :param explainer: pretrained SHAP explainer
    :return: array_like of shape (n_features,)
             The contributions of each feature in x
    """
    x = np.asanyarray(x)
    contributions = explainer.shap_values(x)
    return contributions


