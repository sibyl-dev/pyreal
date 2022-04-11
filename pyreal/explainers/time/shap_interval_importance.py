import numpy as np
import pandas as pd
from shap import Explainer as ShapExplainer
from shap import KernelExplainer, LinearExplainer

from pyreal.explainers import TimeSeriesImportanceBase
from pyreal.types.explanations.dataframe import AdditiveFeatureContributionExplanation


def time_to_feature(dataset, time_column):
    dataset = dataset[time_column]
    pd.DataFrame()
    return dataset

class IntervalImportance(TimeSeriesImportanceBase):
    """
    IntervalImportance object.

    A IntervalImportance object creates features of time-series data by aggregating timestamps into 
    intervals and produce explanation using the SHAP algorithm.

    IntervalImportance explainers expect data in the **model-ready feature space**

    Currently, only classification models explanation is supported.

    Args:
        model (string filepath or model object):
           Filepath to the pickled model to explain, or model object with .predict() function
        x_train_orig (DataFrame of size (n_instances, length of series)):
            Training set in original form.
        interval_size (int):
            The size of the interval.
        shap_type (string, one of ["kernel", "linear"]):
            Type of shap algorithm to use. If None, SHAP will pick one.
        **kwargs: see base Explainer args
    """

    def __init__(self, model, x_train_orig,
                 window_size=1, shap_type=None, **kwargs):
        supported_types = ["kernel", "linear"]
        if shap_type is not None and shap_type not in supported_types:
            raise ValueError("Shap type not supported, given %s, expected one of %s or None" %
                             (shap_type, str(supported_types)))
        else:
            self.shap_type = shap_type

        self.window_size = window_size
        self.explainer = None
        self.explainer_input_size = None
        super(IntervalImportance, self).__init__(model, x_train_orig, **kwargs)

    def fit(self):
        """
        Fit the contribution explainer
        """
        dataset = self.transform_to_x_algorithm(self._x_train_orig)
        self.explainer_input_size = dataset.shape[1]
        if self.shap_type == "kernel":
            self.explainer = KernelExplainer(self.model.predict, dataset)
        # Note: we manually check for linear model here because of SHAP bug
        elif self.shap_type == "linear":
            self.explainer = LinearExplainer(self.model, dataset)
        else:
            self.explainer = ShapExplainer(self.model, dataset)  # SHAP will pick an algorithm
        return self

    def get_contributions(self, x_orig):
        """
        Calculate the explanation of each feature in x using SHAP.

        Args:
            x_orig (DataFrame of shape (n_instances, n_features)):
               The input to be explained
        Returns:
            DataFrame of shape (n_instances, n_features):
                 The contribution of each feature
        """
        if self.explainer is None:
            raise AttributeError("Instance has no explainer. Must call "
                                 "fit() before "
                                 "produce()")
        x = self.transform_to_x_model(x_orig)
        if x.shape[1] != self.explainer_input_size:
            raise ValueError("Received input of wrong size."
                             "Expected ({},), received {}"
                             .format(self.explainer_input_size, x.shape))
        old_columns = x.columns
        x = np.asanyarray(x)

        shap_values = np.array(self.explainer.shap_values(x))   
        if shap_values.ndim < 2:
            raise RuntimeError("Something went wrong with SHAP - expected at least 2 dimensions")

        columns = []
        if x.shape[1] > 1:
            num_windows = (x.shape[1]-1) // self.window_size
            indices = [(i+1)*self.window_size for i in range(num_windows)]
            indices.insert(0,0)
            # Rename columns of explanation
            if self.window_size > 1:
                columns = ['time '+id for id in old_columns]
            columns = [f'time {indices[i]} to {indices[i+1]-1}' for i in range(len(indices)-1)]
            if indices[-1] == len(indices)-1:
                columns.append(f'time {indices[-1]}')
            agg_shap_values = np.add.reduceat(shap_values, indices, axis=1)
        else:
            agg_shap_values = shap_values


        if shap_values.ndim == 2:
            return AdditiveFeatureContributionExplanation(
                pd.DataFrame(agg_shap_values, columns=columns))
        if shap_values.ndim > 2:
            predictions = self.model_predict(x_orig)
            if self.classes is not None:
                predictions = [np.where(self.classes == i)[0][0] for i in predictions]
            shap_values = shap_values[predictions, np.arange(shap_values.shape[1]), :]
            return AdditiveFeatureContributionExplanation(
                pd.DataFrame(agg_shap_values, columns=columns))

# problem: data is susceptible to time-shift.
