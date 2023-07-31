import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.base import is_classifier
from sklearn.metrics import get_scorer

from pyreal.transformers import BreakingTransformError
from pyreal.transformers import fit_transformers as fit_transformers_func
from pyreal.transformers import run_transformers
from pyreal.types.explanations.base import Explanation
from pyreal.utils import model_utils

log = logging.getLogger(__name__)


def _check_transformers(transformers):
    """
    Validate that all Transformers in `transformers` are legal. Converts single Transformer objects
    into lists. Checks for the existence of a `.transform()` function for all Transformers.

    Args:
        transformers (Transformer or list of Transformers):
            A list of Transformer objects to validate
    Returns:
        List of Transformers
            The original input list, or a single Transformer converted to a list

    Raises:
        TypeError
            If one or more objects in `transformers` does not have a `.transform()` function.
    """
    if transformers is None:
        return []
    if not isinstance(transformers, list):
        transformers = [transformers]
    else:
        transformers = transformers
    for transformer in transformers:
        transform_method = getattr(transformer, "transform", None)
        if not callable(transform_method):
            raise TypeError("Given transformer that does not have a .transform function")
    return transformers


def _get_transformers(transformers, algorithm=None, model=None, interpret=None):
    """
    Return Transformers in `transformers` that have all the requested flags.

    Args:
        transformers (list of Transformers):
            List from which to pick transformers
        algorithm (Boolean or None):
            If True or False, choose transformers with that value. If None, do not consider the
            value of this flag.
        model (Boolean or None):
            If True or False, choose transformers with that value. If None, do not consider the
            value of this flag.
        interpret (Boolean or None):
            If True or False, choose transformers with that value. If None, do not consider the
            value of this flag.

    Returns:
        List of Transformers
            A list of Transformers from `transformers` that have all requested flags.
    """
    select_transformers = []
    for t in transformers:
        if (
            (algorithm is None or t.algorithm == algorithm)
            and (model is None or t.model == model)
            and (interpret is None or t.interpret == interpret)
        ):
            select_transformers.append(t)
    return select_transformers


class ExplainerBase(ABC):
    """
    Generic ExplainerBase object

    Args:
        model (string filepath or model object):
           Filepath to the pickled model to explain, or model object with .predict() function
           model.predict() should return a single value prediction for each input
           Classification models should return the index or class. If the latter, the `classes`
           parameter should be provided.
        x_train_orig (DataFrame of shape (n_instances, x_orig_feature_count)):
           The training set for the explainer. If none, must be provided separately when fitting
        y_train (DataFrame of shape (n_instances,)):
           The y values for the dataset
        e_algorithm (string):
            Algorithm to use, if applicable
        feature_descriptions (dict):
           Interpretable descriptions of each feature
        classes (array):
            List of class names returned by the model, in the order that the internal model
            considers them if applicable.
            Can be automatically extracted if model is an sklearn classifier
            None if model is not a classifier
        class_descriptions (dict):
            Interpretable descriptions of each class
            None if model is not a classifier
        transformers (transformer object or list of transformer objects):
           Transformer(s) used by the Explainer.
        fit_on_init (Boolean):
           If True, fit the explainer on initiation.
           If False, self.fit() must be manually called before produce() is called
        training_size (Integer):
            If given this value, sample a training set with size of this value
            from x_train_orig and use it to train the explainer instead of the
            entire x_train_orig.
        return_original_explanation (Boolean):
            If True, return the explanation originally generated without any transformations
        fit_transformers (Boolean):
            If True, fit transformers on x_train_orig. Requires x_train_orig not be None
    """

    def __init__(
        self,
        model,
        x_train_orig=None,
        y_train=None,
        e_algorithm=None,
        feature_descriptions=None,
        classes=None,
        class_descriptions=None,
        transformers=None,
        fit_on_init=False,
        training_size=None,
        return_original_explanation=False,
        fit_transformers=False,
    ):
        if isinstance(model, str):
            self.model = model_utils.load_model_from_pickle(model)
        else:
            predict_method = getattr(model, "predict", None)
            if not callable(predict_method):
                raise TypeError("Given model that does not have a .predict function")
            self.model = model

        self.x_train_orig = x_train_orig
        self.y_train = y_train

        if x_train_orig is not None and not isinstance(x_train_orig, pd.DataFrame):
            raise TypeError("x_train_orig must be of type DataFrame")
        if y_train is not None and not (
            isinstance(y_train, pd.DataFrame) or isinstance(y_train, pd.Series)
        ):
            raise TypeError("y_train must be of type DataFrame or Series")

        self.e_algorithm = e_algorithm

        self.transformers = _check_transformers(transformers)

        self.feature_descriptions = feature_descriptions

        self.classes = classes
        if (
            classes is None
            and (
                str(self.model.__module__).startswith("sklearn")
                or str(self.model.__module__).startswith("sktime")
            )
            and is_classifier(model)
            and hasattr(model, "classes_")
        ):
            self.classes = model.classes_

        self.class_descriptions = class_descriptions
        self.return_original_explanation = return_original_explanation

        self.x_train_orig_subset = self.x_train_orig
        self.y_train_subset = self.y_train

        self.training_size = training_size
        if x_train_orig is not None and training_size is not None:
            self.x_train_orig_subset, self.y_train_subset = self._select_training_set(
                x_train_orig, y_train
            )

        if fit_transformers:
            if x_train_orig is None:
                raise ValueError("Cannot fit transformers unless x_train_orig is provided")
            a_transformers = _get_transformers(self.transformers, algorithm=True)
            i_transformers = _get_transformers(self.transformers, interpret=True)
            fit_transformers_func(a_transformers, self.x_train_orig)
            fit_transformers_func(i_transformers, self.x_train_orig)

        if fit_on_init:
            if x_train_orig is None:
                raise
            self.fit()

    def fit(self, x_train_orig=None, y_train=None):
        """
        Fit this explainer object.

        Args:
            x_train_orig (DataFrame of shape (n_instances, n_features):
                Training set to fit on, required if not provided on initialization
            y_train:
                Targets of training set, required if not provided on initialization
        """
        return self

    @abstractmethod
    def produce(self, x_orig):
        """
        Return the explanation, in the desired form.

        Args:
            x_orig (DataFrame of shape (n_instances, n_features):
                Input to explain

        Returns:
            Type varies by subclass
                Explanation
        """

    def transform_to_x_algorithm(self, x_orig):
        """
        Transform x_orig to x_algorithm, using the algorithm transformers

        Args:
            x_orig (DataFrame of shape (n_instances, x_orig_feature_count)  or Series):
                Original input

        Returns:
             DataFrame or Series of shape (n_instances, x_algorithm_feature_count)
                x_orig converted to explainable form
        """
        a_transformers = _get_transformers(self.transformers, algorithm=True)
        return run_transformers(a_transformers, x_orig)

    def transform_to_x_model(self, x_orig):
        """
        Transform x_orig to x_model, using the model transformers

        Args:
            x_orig (DataFrame of shape (n_instances, x_orig_feature_count) or Series):
                Original input

        Returns:
             DataFrame or Series of shape (n_instances, x_model_feature_count)
                x_orig converted to model-ready form
        """
        m_transformers = _get_transformers(self.transformers, model=True)
        return run_transformers(m_transformers, x_orig)

    def transform_x_from_algorithm_to_model(self, x_algorithm):
        """
        Transform x_algorithm to x_model, using the model transformers

        Args:
            x_algorithm (DataFrame of shape (n_instances, x_orig_feature_count) or Series):
                Input in explain space

        Returns:
             DataFrame or Series of shape (n_instances, x_model_feature_count)
                x_algorithm converted to model-ready form
        """
        m_transformers = _get_transformers(self.transformers, algorithm=False, model=True)
        return run_transformers(m_transformers, x_algorithm)

    def transform_to_x_interpret(self, x_orig):
        """
        Transform x_orig to x_interpret, using the interpret transformers
        Args:
            x_orig (DataFrame of shape (n_instances, x_orig_feature_count) or Series):
                Original input

        Returns:
             DataFrame or Series of shape (n_instances, x_interpret_feature_count)
                x_orig converted to interpretable form
        """
        i_transformers = _get_transformers(self.transformers, interpret=True)
        return run_transformers(i_transformers, x_orig)

    def transform_explanation(self, explanation, x_orig=None):
        """
        Transform the explanation into its interpretable form, by running the e_transformer's
        "inverse_transform_explanation" and i_transformers "transform_explanation" functions.
        If an `x_orig` is provided, also convert `x_orig` with the same
        transformers. This function will result in `values` in the Explanation object
        in the same feature space as the final explanation

        Args:
            explanation (type varies by subclass):
                The raw explanation to transform
            x_orig (DataFrame of shape (n_instances, n_features) or None)
                Data to transform to final space

        Returns:
            Explanation
                The interpretable form of the explanation
        """
        if not isinstance(explanation, Explanation):
            raise ValueError("explanation is not a valid Explanation object")

        convert_x = x_orig is not None
        if self.return_original_explanation:
            if convert_x:
                explanation = explanation.update_values(self.transform_to_x_algorithm(x_orig))
            return explanation

        x = None
        if convert_x:
            x = x_orig.copy()

        a_transformers = _get_transformers(self.transformers, algorithm=True, interpret=False)
        i_transformers = _get_transformers(self.transformers, interpret=True)

        # Iterate through algorithm transformers
        for i, t in enumerate(a_transformers[::-1]):
            try:
                explanation = t.inverse_transform_explanation(explanation)
            # If this is a breaking transformer, transform x to the current point and return
            except BreakingTransformError:
                log.warning(
                    "Transformer class %s does not have the required inverse explanation "
                    "transform and is set to break, stopping transform process"
                    % type(t).__name__
                )
                break_point = len(a_transformers) - i
                if convert_x:
                    x = run_transformers(a_transformers[0:break_point], x)
                    return explanation.update_values(x)
                else:
                    return explanation
        # Iterate through interpret transformers
        for t in i_transformers:
            if not t.algorithm:
                try:
                    explanation = t.transform_explanation(explanation)
                except BreakingTransformError:
                    log.warning(
                        "Transformer class %s does not have the required explanation "
                        "transform and is set to break, stopping transform process"
                        % type(t).__name__
                    )
                    if convert_x:
                        return explanation.update_values(x)
                    return explanation
            if convert_x:
                x = t.transform(x)
        if convert_x:
            return explanation.update_values(x)
        return explanation

    def model_predict(self, x_orig):
        """
        Predict on x_orig using the model and return the result

        Args:
            x_orig (DataFrame of shape (n_instances, x_orig_feature_count)):
                Data to predict on

        Returns:
            DataFrame of shape (n_instances,)
                Model prediction on x_orig
        """
        if x_orig.ndim == 1:
            x_orig = x_orig.to_frame().T
        x_model = self.transform_to_x_model(x_orig)
        return self.model.predict(x_model)

    def model_predict_on_algorithm(self, x_algorithm):
        """
        Predict on x_algorithm using the model and return the result

        Args:
            x_algorithm (DataFrame of shape (n_instances, x_orig_feature_count)):
                Data to predict on

        Returns:
            DataFrame of shape (n_instances,)
                Model prediction on x_orig
        """
        if x_algorithm.ndim == 1:
            x_algorithm = x_algorithm.to_frame().T
        x_model = self.transform_x_from_algorithm_to_model(x_algorithm)
        return self.model.predict(x_model)

    def feature_description(self, feature_name):
        """
        Returns the interpretable description associated with a feature

        Args:
            feature_name (string)

        Returns:
            string
                 Description of feature
        """
        return self.feature_descriptions[feature_name]

    def convert_columns_to_interpretable(self, df):
        """
        Returns df with columns (or index, for series) converted to the interpretable descriptions

        Args:
            df (DataFrame):

        Returns:
            string
                 Description of feature
        """
        if self.feature_descriptions is None:
            return df

        if isinstance(df, pd.Series):
            return df.rename(self.feature_descriptions)
        return df.rename(self.feature_descriptions, axis="columns")

    def convert_data_to_interpretable(self, x_orig):
        """
        Convert data in its original form to an interpretable form, with interpretable features
        Args:
            x_orig (DataFrame of shape (n_instances, n_features) or Series):
                Input data to convert

        Returns:
            DataFrame of shape (n_instances, x_interpret_feature_count)
                Transformed, interpretable data
        """
        return self.convert_columns_to_interpretable(self.transform_to_x_interpret(x_orig))

    def evaluate_model(self, scorer, x_orig=None, y=None):
        """
        Evaluate the model using a chosen scorer algorithm.

        Args:
            scorer (string):
                Type of scorer to use. See sklearn's scoring parameter options here:
                https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
            x_orig (DataFrame of shape (n_instances, n_features)):
                Dataset to score on. Required if x_train_orig was not provided at initialization.
                If None, use self.x_train_orig
            y (DataFrame of shape (n_instances, n_features)):
                Dataset to score on. Required if y_train was not provided at initialization


        Returns:
            float
                A score for the model
        """
        x_orig, y = self._get_training_data(x_orig, y)

        scorer = get_scorer(scorer)
        x_model = self.transform_to_x_model(x_orig)
        score = scorer(self.model, x_model, y)
        return score

    @abstractmethod
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

    def _get_x_train_orig(self, x_train_orig):
        """
        Helper function to get the appropriate x_orig or raise errors if something goes wrong
        Args:
            x_train_orig (DataFrame or None):
                Provided DataFrame
        Returns:
            The dataframe to use (x_orig or self.x_train_orig)

        Raises:
            ValueError if no valid dataframe
        """
        if x_train_orig is not None:
            return self._select_training_set(x_train_orig)[0]
        if self.x_train_orig_subset is not None:
            return self.x_train_orig_subset
        else:
            raise ValueError("Must provide x_train_orig at initialization or fitting time!")

    def _get_training_data(self, x_train_orig, y_train):
        if x_train_orig is None and self.x_train_orig is None:
            raise ValueError("Must provide x_train_orig at initialization or fitting time")
        if y_train is None and self.y_train is None:
            raise ValueError("Must provide y_train at initialization or fitting time")
        if x_train_orig is not None and y_train is None:
            raise ValueError("Must provide y_train if providing x_train_orig")

        if x_train_orig is None:
            return self.x_train_orig_subset, self.y_train_subset
        else:
            return self._select_training_set(x_train_orig, y_train)

    def _select_training_set(self, x_train, y_train=None):
        if self.training_size is None:
            return x_train, y_train
        if self.training_size < len(x_train.index):
            if self.classes is not None and self.training_size < len(self.classes):
                raise ValueError("training_size must be larger than the number of classes")
            else:
                data_sample_indices = pd.Index(
                    np.random.choice(x_train.index, self.training_size, replace=False)
                )

            # use x_train_orig_subset for fitting explainer
            x_train_subset = x_train.loc[data_sample_indices]
            if y_train is not None:
                y_train_subset = y_train.loc[data_sample_indices]
                return x_train_subset, y_train_subset
            return x_train_subset, None
        else:
            return x_train, y_train
