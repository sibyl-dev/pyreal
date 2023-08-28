from pyreal.explainers import ExplainerBase, ShapFeatureContribution, ShapFeatureImportance


class Explainer(ExplainerBase):
    """
    A generic Explainer wrapper.

    A GenericExplainer object assigns a default ML explainer based on the type of x_orig.

    Args:
        model (string filepath or model object):
           Filepath to the pickled model to explain, or model object with .predict() function
        x_train_orig (dataframe of shape (n_instances, x_orig_feature_count)):
           The training set for the explainer
        scope (string of either "global" or "local"):
            Whether the explainer is global or local
        e_algorithm (string, one of ["shap"]):
           Explanation algorithm to use. If none, one will be chosen automatically based on model
           type
        interpretable_features (Boolean):
            If True, return explanations using the interpretable feature descriptions instead of
            default names
        **kwargs: see base Explainer args
    """

    def __init__(
        self,
        model,
        x_train_orig=None,
        scope="global",
        e_algorithm="shap",
        interpretable_features=True,
        **kwargs
    ):
        self.scope = scope
        self.interpretable_features = interpretable_features
        algorithm_list = ["shap"]

        super(Explainer, self).__init__(model, x_train_orig, **kwargs)
        # TODO: implement smart choosing algorithm based on type of x
        if e_algorithm not in algorithm_list:
            raise ValueError("Invalid algorithm type %s" % e_algorithm)
        if scope == "global":
            if e_algorithm == "shap":
                self.base_explainer = ShapFeatureImportance(
                    model, x_train_orig, interpretable_features=interpretable_features, **kwargs
                )
        elif scope == "local":
            if e_algorithm == "shap":
                self.base_explainer = ShapFeatureContribution(
                    model, x_train_orig, interpretable_features=interpretable_features, **kwargs
                )
        elif scope == "testing":  # Only to be used for unit testing purposes
            self.base_explainer = None
        else:
            raise TypeError("Explainers must be either global or local")

    def fit(self, x_train_orig=None, y_train=None):
        """
        Fit this explainer object

        Args:
             x_train_orig (DataFrame of shape (n_instances, n_features):
                Training set to fit on, required if not provided on initialization
            y_train:
                Targets of training set, required if not provided on initialization
        """
        self.base_explainer.fit(x_train_orig, y_train)

    def produce_explanation(self, x_orig=None, **kwargs):
        """
        Produce the explanation

        Args:
            x_orig (DataFrame of shape (n_instances, n_features)):
                Input to explain

        Returns:
            DataFrame of shape (n_instances, n_features)
                Contribution of each feature for each instance
        """
        if self.scope == "global" and x_orig is not None:
            raise ValueError(
                "Global explainer does not explain specific input data.                           "
                "    Call produce() without arguments or change scope to local."
            )
        if self.scope == "local" and x_orig is None:
            raise ValueError(
                "Local explainers requires input data.                               Call"
                " produce() with input or change scope to global."
            )

        return self.base_explainer.produce_explanation(x_orig)

    def evaluate_variation(self, with_fit=False, explanations=None, n_iterations=20, n_rows=10):
        """
        Evaluate the variation of the explanations generated by this Explainer.
        A variation of 0 means this explainer is expected to generate the exact same explanation
        given the same model and input. Variation is always non-negative, and can be arbitrarily
        high.

        Args:
            with_fit (Boolean):
                If True, evaluate the variation in explanations including the fit (fit each time
                before running). If False, evaluate the variation in explanations of a pre-fit
                Explainer.
            explanations (None or List of Explanation Objects):
                If provided, run the variation check on the precomputed list of explanations
                instead of generating
            n_iterations (int):
                Number of explanations to generate to evaluation variation
            n_rows (int):
                Number of rows of dataset to generate explanations on

        Returns:
            float
                The variation of this Explainer's explanations
        """
        return self.base_explainer.evaluate_variation(
            with_fit=with_fit, explanations=explanations, n_iterations=n_iterations, n_rows=n_rows
        )
