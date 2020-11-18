import os
import pickle
import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from sklearn.linear_model import LinearRegression

from pyreal.explainers import LocalFeatureContribution
from pyreal.utils.transformer import OneHotEncoderWrapper

TEST_ROOT = os.path.dirname(os.path.abspath(__file__))


class TestFeatureExplanation(unittest.TestCase):
    """Tests for `pyreal` package."""

    def setUp(self):
        """Set up test fixtures"""
        try:
            os.makedirs(os.path.join(TEST_ROOT, "data"))
        except FileExistsError:
            pass

        self.X_train = pd.DataFrame([[2, 1, 3],
                                     [4, 3, 4],
                                     [6, 7, 2]], columns=["A", "B", "C"])
        self.y_train = pd.DataFrame([2, 4, 6])
        self.expected = np.mean(self.y_train)[0]
        model_no_transforms = LinearRegression()
        model_no_transforms.fit(self.X_train, self.y_train)
        model_no_transforms.coef_ = np.array([1, 0, 0])
        model_no_transforms.intercept_ = 0
        self.model_no_transforms_filename = os.path.join(TEST_ROOT, "data",
                                                         "model_no_transforms.pkl")
        with open(self.model_no_transforms_filename, "wb") as f:
            pickle.dump(model_no_transforms, f)

        # TODO: replace with ML primitives
        self.one_hot_encoder = OneHotEncoderWrapper(feature_list=["A"])
        self.one_hot_encoder.fit(self.X_train)
        self.X_transformed = self.one_hot_encoder.transform(self.X_train)
        self.y_transformed = pd.DataFrame([1, 2, 3])
        self.expected_transformed = np.mean(self.y_transformed)[0]
        model_one_hot = LinearRegression()
        model_one_hot.fit(self.X_transformed, self.y_transformed)
        model_one_hot.coef_ = np.array([0, 0, 1, 2, 3])
        model_one_hot.intercept_ = 0
        self.model_one_hot_filename = os.path.join(TEST_ROOT, "data", "model_one_hot.pkl")
        with open(self.model_one_hot_filename, "wb") as f:
            pickle.dump(model_one_hot, f)

    def tearDown(self):
        """Tear down test fixtures"""
        os.remove(self.model_no_transforms_filename)
        os.remove(self.model_one_hot_filename)

    def test_init_invalid_transforms(self):
        invalid_transform = "invalid"
        with self.assertRaises(TypeError):
            LocalFeatureContribution(self.model_no_transforms_filename, self.X_train,
                                     m_transforms=invalid_transform)
        with self.assertRaises(TypeError):
            LocalFeatureContribution(self.model_no_transforms_filename, self.X_train,
                                     e_transforms=invalid_transform)
        with self.assertRaises(TypeError):
            LocalFeatureContribution(self.model_no_transforms_filename, self.X_train,
                                     i_transforms=invalid_transform)

    def test_init_invalid_model(self):
        invalid_model = []
        with self.assertRaises(TypeError):
            LocalFeatureContribution(invalid_model, self.X_train)

    def test_run_transforms(self):
        expected = pd.DataFrame([[1, 3, 1, 0, 0],
                                 [3, 4, 0, 1, 0],
                                 [7, 2, 0, 0, 1]], columns=["B", "C", "A_2", "A_4", "A_6"])
        explainer = LocalFeatureContribution(self.model_no_transforms_filename, self.X_train,
                                             e_transforms=self.one_hot_encoder,
                                             m_transforms=self.one_hot_encoder,
                                             i_transforms=self.one_hot_encoder)
        result = explainer.transform_to_x_interpret(self.X_train)
        assert_frame_equal(result, expected, check_like=True, check_dtype=False)
        result = explainer.transform_to_x_model(self.X_train)
        assert_frame_equal(result, expected, check_like=True, check_dtype=False)
        result = explainer.transform_to_x_explain(self.X_train)
        assert_frame_equal(result, expected, check_like=True, check_dtype=False)

    def test_predict(self):
        explainer = LocalFeatureContribution(self.model_no_transforms_filename, self.X_train)
        expected = [2, 4, 6]
        result = explainer.model_predict(self.X_train)
        self.assertTrue(np.array_equal(result, expected))

        explainer = LocalFeatureContribution(self.model_one_hot_filename, self.X_train,
                                             m_transforms=self.one_hot_encoder)
        expected = [1, 2, 3]
        result = explainer.model_predict(self.X_train)
        self.assertTrue(np.array_equal(result, expected))
