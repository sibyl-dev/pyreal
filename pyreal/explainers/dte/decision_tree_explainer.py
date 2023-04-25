from pyreal.explainers import DecisionTreeExplainerBase, SurrogateDecisionTree


def choose_algorithm():
    """
    Choose an algorithm based on the model type.
    Currently, shap is the only supported algorithm

    Return:
        string (one of ["surrogate_tree"])
            Explanation algorithm to use
    """
    return "surrogate_tree"


class DecisionTreeExplainer(DecisionTreeExplainerBase):
    """
    Generic DecisionTreeExplainer wrapper

    An DecisionTreeExplainer object wraps multiple decision tree-based explanations. If no
    specific algorithm is requested, one will be chosen based on the information given.
    Currently, only surrogate tree is supported.

    Args:
        model (string filepath or model object):
           Filepath to the pickled model to explain, or model object with .predict() function
        x_train_orig (dataframe of shape (n_instances, x_orig_feature_count)):
           The training set for the explainer
        e_algorithm (string, one of ["surrogate_tree"]):
           Explanation algorithm to use. If none, one will be chosen automatically based on model
           type
        is_classifier (bool):
            Set this True for a classification model, False for a regression model.
        max_depth (int):
            The max_depth of the tree
        **kwargs: see DecisionTreeExplainerBase args
    """

    def __init__(
        self,
        model,
        x_train_orig=None,
        e_algorithm=None,
        is_classifier=True,
        max_depth=None,
        **kwargs
    ):
        self.is_classifier = is_classifier
        self.max_depth = max_depth
        if e_algorithm is None:
            e_algorithm = choose_algorithm()
        if e_algorithm == "surrogate_tree":
            self.base_decision_tree = SurrogateDecisionTree(
                model, x_train_orig, is_classifier, max_depth, **kwargs
            )
        if self.base_decision_tree is None:
            raise ValueError("Invalid algorithm type %s" % e_algorithm)

        super(DecisionTreeExplainer, self).__init__(model, x_train_orig, **kwargs)

    def fit(self, x_train_orig=None, y_train=None):
        """
        Fit this explainer object

        Args:
             x_train_orig (DataFrame of shape (n_instances, n_features):
                Training set to fit on, required if not provided on initialization
            y_train:
                Targets of training set, required if not provided on initialization
        """
        self.base_decision_tree.fit(x_train_orig, y_train)
        return self

    def produce(self, x_orig=None):
        """
        Returns the decision tree object, either DecisionTreeClassifier or DecisionTreeRegressor

        x_orig is a dummy param to match signature
        """
        return self.base_decision_tree.produce()

    def produce_importances(self):
        """
        Returns the feature importance created by the decision tree explainer
        """
        return self.base_decision_tree.produce_importances()
