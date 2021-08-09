from sklearn import tree
# from sklearn.model_selection import GridSearchCV

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
        is_classifier (bool):
            Set this True for a classification model, False for a regression model.
        **kwargs: see base Explainer args
    """

    def __init__(self, model, x_train_orig, is_classifier=True, max_depth=None, **kwargs):
        self.explainer = None
        self.algorithm = ExplanationAlgorithm.SURROGATE_DECISION_TREE
        self.explainer_input_size = None
        self.is_classifer = is_classifier
        self.max_depth = max_depth
        super(SurrogateDecisionTree, self).__init__(self.algorithm, model, x_train_orig, **kwargs)

    def fit(self):
        """
        Fit the decision tree.
        TODO: Perhaps use sklearn's GridSearchCV to find the "best" tree.
        """
        e_dataset = self.transform_to_x_explain(self.x_train_orig)
        m_dataset = self.transform_to_x_model(self.x_train_orig)
        self.explainer_input_size = e_dataset.shape[1]
        if self.is_classifer:
            self.explainer = tree.DecisionTreeClassifier(max_depth=self.max_depth)
            self.explainer.fit(e_dataset, self.model.predict(m_dataset))
        else:
            self.explainer = tree.DecisionTreeRegressor(max_depth=self.max_depth)
            self.explainer.fit(e_dataset, self.model.predict(m_dataset))

    def produce(self):
        """
        Produce the explanation as a decision tree model.

        Returns:
            An explanation class in the form of a decision tree model
        """

        if self.explainer is None:
            self.fit()
            # raise AttributeError("Instance has no explainer. Decision tree training failed.")

        return self.explainer
