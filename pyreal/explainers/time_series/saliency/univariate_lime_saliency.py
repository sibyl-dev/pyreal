import warnings
from contextlib import ExitStack

import numpy as np
import pandas as pd
from lime import lime_tabular

from pyreal.explainers.time_series import SaliencyBase
from pyreal.types.explanations.feature_based import FeatureContributionExplanation


class UnivariateLimeSaliency(SaliencyBase):
    """
    UnivariateLimeSaliency object.

    An UnivariateLimeSaliency object judges the relative importance or saliency of each timestep
    value using the LIME algorithm. Currently only supports classification models.

    (Ribeiro, M. T., Singh, S., & Guestrin, C. (2016).
    “Why Should I Trust You?”: Explaining the Predictions of Any Classifier.
    Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge
    Discovery and Data Mining. http://arxiv.org/abs/1602.04938)

    Can only take a single row input to .produce()
    """

    def __init__(
        self, model, x_train_orig, y_orig, regression=False, suppress_prob_warnings=False, **kwargs
    ):
        """
        Args:
            model (string filepath or model object):
                Filepath to the pickled model to explain, or model object with .predict() function
            x_train_orig (DataFrame of size (n_instances, n_features)):
                Training set in original form.
            y_orig (DataFrame of shape (n_instances,)):
                The y values for the dataset
            regression (Boolean):
                If true, model is a regression model.
                If false, must provide a num_classes or classes parameter
            suppress_prob_warnings (Boolean):
                LIME warns when class predictions do not sum to 1, because it suggests the model
                is not predicting probabilites. In some cases, such as multilabel prediction,
                this warning may be incorrect. In this case, set this parameter to True.
            **kwargs: see base Explainer args
        """
        self.suppress_prob_warnings = suppress_prob_warnings
        self.explainer = None
        self.regression = regression
        super(UnivariateLimeSaliency, self).__init__(model, x_train_orig, y_orig=y_orig, **kwargs)

    def fit(self):
        x_train_algo = self.transform_to_x_algorithm(self.x_train_orig)
        num_timesteps = x_train_algo.shape[1]

        x_train_algo_np = np.copy(x_train_algo)[: self.training_size, :]
        y_train_np = np.copy(self.y_orig)[: self.training_size]

        if self.regression:
            mode = "regression"
        else:
            mode = "classification"

        self.explainer = lime_tabular.RecurrentTabularExplainer(
            np.expand_dims(x_train_algo_np, 2),
            training_labels=y_train_np,
            feature_names=np.arange(num_timesteps),
            class_names=self.class_descriptions,
            categorical_features=np.arange(num_timesteps),
            mode=mode,
        )

        return self

    def get_contributions(self, x_orig):
        """
        Calculate the explanation of each feature in x using LIME.
        Args:
            x_orig (DataFrame of shape (n_instances, n_features)):
               The input to be explained
        Returns:
            DataFrame of shape (n_instances, n_features):
                 The contribution of each feature
        """
        if self.explainer is None:
            raise AttributeError(
                "Instance has no lime explainer. Must call fit() before produce()"
            )

        x_algo = self.transform_to_x_algorithm(x_orig)
        num_timesteps = x_algo.shape[1]

        with ExitStack() as stack:
            if self.suppress_prob_warnings:
                stack.enter_context(warnings.catch_warnings())
                warnings.simplefilter("ignore")
            if self.classes is not None:
                exp = self.explainer.explain_instance(
                    np.array(x_algo), classifier_fn=self.model.predict, labels=self.classes
                )
            else:
                exp = self.explainer.explain_instance(
                    np.array(x_algo), classifier_fn=self.model.predict
                )
        explanation = exp.as_map()

        # Convert the lime explanation format to a DataFrame
        importances = {}
        # if this is a regression model, just return the explanation for positive
        if self.regression:
            classes = [1]
        elif self.classes is not None:
            classes = self.classes
        else:
            classes = list(explanation.keys())
        for class_name in classes:
            class_explanation = dict(explanation[class_name])
            importances[class_name] = np.array(
                [
                    class_explanation[i] if (i in class_explanation) else 0
                    for i in range(num_timesteps)
                ]
            )

        if isinstance(x_algo, pd.DataFrame):
            importances_df = pd.DataFrame(importances, index=x_algo.columns).T
        else:
            importances_df = pd.DataFrame(importances).T

        return FeatureContributionExplanation(importances_df)
