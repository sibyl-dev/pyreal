import pandas as pd
from sklearn import tree

from pyreal.explainers import DecisionTreeExplainerBase
from pyreal.types.explanations.decision_tree import DecisionTreeExplanation
from pyreal.types.explanations.feature_based import FeatureImportanceExplanation


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
        is_classifier (bool):
            Set this True for a classification model, False for a regression model.
        max_depth (int):
            The max_depth of the tree.
        **kwargs: see base Explainer args
    """

    def __init__(self, model, x_train_orig=None, is_classifier=True, max_depth=None, **kwargs):
        self.explainer = None
        self.explainer_input_size = None
        self.is_classifer = is_classifier
        self.max_depth = max_depth
        super(SurrogateDecisionTree, self).__init__(model, x_train_orig, **kwargs)

    def fit(self, x_train_orig=None, y_train=None):
        """
        Fit the decision tree.
        TODO: Perhaps use sklearn's GridSearchCV to find the "best" tree.

        Args:
             x_train_orig (DataFrame of shape (n_instances, n_features):
                Training set to fit on, required if not provided on initialization
            y_train:
                Targets of training set, required if not provided on initialization
        """
        x_train_orig = self._get_x_train_orig(x_train_orig)

        a_dataset = self.transform_to_x_algorithm(x_train_orig)
        m_dataset = self.transform_to_x_model(x_train_orig)

        self.explainer_input_size = a_dataset.shape[1]
        if self.is_classifer:
            self.explainer = tree.DecisionTreeClassifier(max_depth=self.max_depth)
            self.explainer.fit(a_dataset, self.model.predict(m_dataset))
        else:
            self.explainer = tree.DecisionTreeRegressor(max_depth=self.max_depth)
            self.explainer.fit(a_dataset, self.model.predict(m_dataset))
        return self

    def produce(self, x_orig=None):
        """
        Produce the explanation as a decision tree model.

        Returns:
            An explanation class in the form of a decision tree model
        """

        if self.explainer is None:
            raise AttributeError(
                "Instance has no explainer. Please fit the explainer before producing"
                " explanations."
            )

        return DecisionTreeExplanation(self.explainer)

    def produce_importances(self):
        """
        Produce the explanation in terms of feature importances.

        Returns:
            The feature importances of the decision tree explainer.
        """
        if self.explainer is None:
            raise AttributeError(
                "Instance has no explainer. Please fit the explainer             before producing"
                " explanations."
            )

        features = self.return_features()
        importances = pd.DataFrame(self.explainer.feature_importances_[None, :], columns=features)
        return FeatureImportanceExplanation(importances)
