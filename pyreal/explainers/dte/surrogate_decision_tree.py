from pyreal.explainers.dte.decision_tree_explainer import DecisionTreeExplainer
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import scale
from sklearn.linear_model import LogisticRegression, ElasticNet, LinearRegression

from pyreal.explainers import DecisionTreeExplainerBase
from pyreal.utils.transformer import ExplanationAlgorithm

class SurrogateDecisionTree(DecisionTreeExplainerBase):
    """
    SurrogateDecisionTree object.

    A SurrogateDecisionTree object gets feature explanation using the surrogate
    tree algorithm.

    Args:
        model (string filepath or model object):
           Filepath to the pickled model to explain, or model object with .predict() function
        x_train_orig (DataFrame of size (n_instances, n_features)):
            Training set in original form.
        **kwargs: see base Explainer args
    """

    def __init__(self, model, x_train_orig, **kwargs):
        self.explainer = None
        self.algorithm = ExplanationAlgorithm.SHAP
        self.explainer_input_size = None
        super(SurrogateDecisionTree, self).__init__(self.algorithm, model, x_train_orig, **kwargs)

    def fit(self):
        """
        Fit the contribution explainer
        """
        dataset = self.transform_to_x_explain(self.x_train_orig)
        self.explainer_input_size = dataset.shape[1]


    def get_importance(self):
        """
        Calculate the explanation of each feature using SHAP.

        Returns:
            DataFrame of shape (n_features, ):
                 The global importance of each feature
        """
        if self.explainer is None:
            raise AttributeError("Instance has no explainer. Must call "
                                 "fit_contribution_explainer before "
                                 "get_contributions")
        x = self.transform_to_x_explain(self.x_train_orig)
        columns = x.columns
        x = np.asanyarray(x)
        # shap_values = np.array(self.explainer.shap_values(x))

        # if shap_values.ndim < 2:
        #     raise RuntimeError("Something went wrong with SHAP - expected at least 2 dimensions")
        # if shap_values.ndim > 2:
        #     predictions = self.model_predict(x)
        #     if self.classes is not None:
        #         predictions = [np.where(self.classes == i)[0][0] for i in predictions]
        #     shap_values = shap_values[predictions, np.arange(shap_values.shape[1]), :]

        # importances = np.mean(np.absolute(shap_values), axis=0).reshape(1, -1)
        # return pd.DataFrame(importances, columns=columns)
