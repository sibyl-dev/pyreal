from abc import ABC, abstractmethod
from sibyl.utils import model_utils


class Explainer(ABC):
    def __init__(self, model_pickle_filepath,
                 X_orig, y_orig=None,
                 feature_descriptions=None,
                 e_transforms=None, m_transforms=None, i_transforms=None,
                 fit_on_init=False):
        """
        Initialize an Explainer object
        :param model_pickle_filepath: filepath
               Filepath to the pickled model to explain
        :param X_orig: dataframe of shape (n_instances, x_orig_feature_count)
               The training set for the explainer
        :param y_orig: dataframe of shape (n_instances,)
               The y values for the dataset
        :param feature_descriptions: dict
               Interpretable descriptions of each feature
        :param e_transforms: transformer object or list of transformer objects
               Transformer(s) that need to be used on x_orig for the explanation algorithm:
                    x_orig -> x_explain
        :param m_transforms: transformer object or list of transformer objects
               Transformer(s) needed on x_orig to make predictions on the dataset with model, if different
               than ex_transforms
                    x_orig -> x_model
        :param i_transforms: transformer object or list of transformer objects
               Transformer(s) needed to make x_orig interpretable
                    x_orig -> x_interpret
        :param fit_on_init: Boolean
               If True, fit the explainer on initiation.
               If False, self.fit() must be manually called before produce() is called
        """
        # TODO: check is model has .predict function
        # TODO: check if transformer(s) have transform
        # TODO: add multiple different types of model reading utilities, and select one

        self.model = model_utils.load_model_from_pickle(model_pickle_filepath)

        self.X_orig = X_orig
        self.y_orig = y_orig

        self.expected_feature_number = X_orig.shape[1]

        self.x_orig_feature_count = X_orig.shape[1]

        self.i_transforms = i_transforms
        self.m_transforms = m_transforms
        self.e_transforms = e_transforms
        if not isinstance(e_transforms, list):
            self.e_transformers = [e_transforms]
        else:
            self.e_transformers = e_transforms
        if not isinstance(m_transforms, list):
            self.e_transformers = [m_transforms]
        else:
            self.e_transformers = m_transforms
        if not isinstance(i_transforms, list):
            self.e_transformers = [i_transforms]
        else:
            self.e_transformers = i_transforms

        self.feature_descriptions = feature_descriptions

        if fit_on_init:
            self.fit()

    @abstractmethod
    def fit(self):
        """
        Fit this explainer object
        :return: None
        """
        pass

    @abstractmethod
    def produce(self, x):
        """
        Return the explanation
        :return:
        """
        pass

    def transform_to_x_explain(self, x_orig):
        """
        Transform x_orig to x_explain, using the e_transforms
        :param x_orig: DataFrame of shape (n_instances, x_orig_feature_count)
        :return: x_explain: DataFrame of shape (n_instances, x_explain_feature_count)
        """
        if self.e_transforms is None:
            return x_orig
        x_explain = x_orig.copy()
        for transform in self.e_transforms:
            x_explain = transform.transform(x_explain)
        return x_explain

    def transform_to_x_model(self, x_orig):
        """
        Transform x_orig to x_model, using the e_transforms
        :param x_orig: DataFrame of shape (n_instances, x_orig_feature_count)
        :return: x_model: DataFrame of shape (n_instances, x_model_feature_count)
        """
        if self.m_transforms is None:
            return x_orig
        x_model = x_orig.copy()
        for transform in self.m_transforms:
            print(x_model.shape)
            x_model = transform.transform(x_model)
        return x_model

    def transform_to_x_interpret(self, x_orig):
        """
        Transform x_orig to x_interpret, using the e_transforms
        :param x_orig: DataFrame of shape (n_instances, x_orig_feature_count)
        :return: x_interpret: DataFrame of shape (n_instances, x_interpret_feature_count)
        """
        if self.i_transforms is None:
            return x_orig
        x_interpret = x_orig.copy()
        for transform in self.i_transforms:
            x_interpret = transform.transform(x_interpret)
        return x_interpret

    def model_predict(self, x_orig):
        """
        Predict on x_orig using the model and return the result
        :param x_orig: DataFrame of shape (n_instances, x_orig_feature_count)
        :return: prediction
        """
        x_model = self.transform_to_x_model(x_orig)
        return self.model.predict(x_model)

    def feature_description(self, feature_name):
        """
        Returns the interpretable description associated with a feature
        :param feature_name: string
        :return: string
                 Description of feature
        """
        return self.feature_descriptions[feature_name]


