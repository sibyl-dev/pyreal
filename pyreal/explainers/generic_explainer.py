from pyreal.explainers import ExplainerBase, ShapFeatureContribution, ShapFeatureImportance


def explain(
    return_explanation=True,
    return_explainer=True,
    explainer=None,
    model=None,
    x_orig=None,
    x_train_orig=None,
    feature_descriptions=None,
    e_transforms=None,
    m_transforms=None,
    i_transforms=None,
    interpretable_features=True,
):
    """
    Get an explanation of the model for x_orig (default explainer produces global explanations)

    Args:
        return_explanation (Boolean):
            If true, return explanation of features in x_input.
            If true, requires `x_input` and one of `explainer` or (`model and x_train`)
        return_explainer (Boolean):
            If true, return the fitted Explainer object.
            If true, requires one of `explainer` or (`model and x_train`)
        explainer (Explainer):
            Fitted explainer object.
        model (string filepath or model object):
           Filepath to the pickled model to explain, or model object with .predict() function
        x_orig (dataframe of shape (n_instances, x_orig_feature_count)):
           The input to explain
        x_train_orig (dataframe of shape (n_instances, x_orig_feature_count)):
           The training set for the explainer
        feature_descriptions (dict):
           Interpretable descriptions of each feature
        e_transforms (transformer object or list of transformer objects):
           Transformer(s) that need to be used on x_orig for the explanation algorithm:
           x_orig -> x_explain
        m_transforms (transformer object or list of transformer objects):
           Transformer(s) needed on x_orig to make predictions on the dataset with model,
           if different than e_transformers
           x_orig -> x_model
        i_transforms (transformer object or list of transformer objects):
           Transformer(s) needed to make x_orig interpretable
           x_orig -> x_interpret
        interpretable_features (Boolean):
            If True, return explanations using the interpretable feature descriptions instead of
            default names

    Returns:
        Explainer:
            The fitted explainer. Only returned in return_explainer is True
        DataFrame of shape (n_instances, n_features):
            The contribution of each feature. Only returned if return_contributions is True
    """
    if not return_explanation and not return_explainer:
        raise ValueError("return_explanation and return_explainer cannot both be false")
    if explainer is None and (model is None or x_train_orig is None):
        raise ValueError("You must provide either explainer OR model and corresponding x_train")

    if explainer is None:
        explainer = Explainer(
            model,
            x_train_orig,
            feature_descriptions=feature_descriptions,
            e_transforms=e_transforms,
            m_transforms=m_transforms,
            i_transforms=i_transforms,
            interpretable_features=interpretable_features,
            fit_on_init=True,
        )
    if return_explainer and return_explanation:
        return explainer, explainer.produce(x_orig)
    if return_explainer:
        return explainer
    if return_explanation:
        return explainer.produce(x_orig)


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
        x_train_orig,
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
        else:
            raise TypeError("Explainers must be either global or local")

    def fit(self):
        """
        Fit this explainer object
        """
        self.base_explainer.fit()

    def produce(self, x_orig=None):
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

        return self.base_explainer.produce(x_orig)

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
