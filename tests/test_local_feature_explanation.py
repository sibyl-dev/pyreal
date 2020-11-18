import os
import pickle
import unittest

import numpy as np
import pandas as pd
from shap import LinearExplainer
from sklearn.linear_model import LinearRegression

from pyreal.explainers import LocalFeatureContribution, ShapFeatureContribution
from pyreal.utils.transformer import OneHotEncoderWrapper

TEST_ROOT = os.path.dirname(os.path.abspath(__file__))


def identity(x):
    return x


class TestFeatureExplanation(unittest.TestCase):
    """Tests for `pyreal` package."""

    def setUp(self):
        """Set up test fixtures"""
        try:
            os.makedirs(os.path.join(TEST_ROOT, "../test/data"))
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
        self.model_one_hot_filename = os.path.join(TEST_ROOT, "data", "model_one_hot.pkl")
        with open(self.model_one_hot_filename, "wb") as f:
            pickle.dump(model_one_hot, f)

    def tearDown(self):
        """Tear down test fixtures"""
        os.remove(self.model_no_transforms_filename)
        os.remove(self.model_one_hot_filename)

    def test_fit_shap_no_transforms(self):
        lfc_object = LocalFeatureContribution(
            model=self.model_no_transforms_filename,
            x_orig=self.X_train, e_algorithm='shap')
        lfc_object.fit()

        shap = ShapFeatureContribution(
            model=self.model_no_transforms_filename, x_orig=self.X_train)
        shap.fit()
        self.assertIsNotNone(shap.explainer)
        self.assertIsInstance(shap.explainer, LinearExplainer)

    def test_produce_shap_no_transforms(self):
        lfc = LocalFeatureContribution(model=self.model_no_transforms_filename,
                                       x_orig=self.X_train, e_algorithm='shap',
                                       fit_on_init=True)
        shap = ShapFeatureContribution(
            model=self.model_no_transforms_filename, x_orig=self.X_train,
            fit_on_init=True)

        self.helper_produce_shape_no_transforms(lfc)
        self.helper_produce_shape_no_transforms(shap)

    def helper_produce_shape_no_transforms(self, explainer):
        x_one_dim = pd.DataFrame([[10, 10, 10]], columns=["A", "B", "C"])
        x_multi_dim = pd.DataFrame([[10, 1, 1],
                                    [0, 2, 3]], columns=["A", "B", "C"])
        contributions = explainer.produce(x_one_dim)
        self.assertEqual(x_one_dim.shape, contributions.shape)
        self.assertEqual(contributions.iloc[0, 0], x_one_dim.iloc[0, 0] - self.expected)
        self.assertEqual(contributions.iloc[0, 1], 0)
        self.assertEqual(contributions.iloc[0, 2], 0)

        contributions = explainer.produce(x_multi_dim)
        self.assertEqual(x_multi_dim.shape, contributions.shape)
        self.assertAlmostEqual(contributions.iloc[0, 0], x_multi_dim.iloc[0, 0] - self.expected)
        self.assertAlmostEqual(contributions.iloc[1, 0], x_multi_dim.iloc[1, 0] - self.expected)
        self.assertTrue((contributions.iloc[:, 1] == 0).all())
        self.assertTrue((contributions.iloc[:, 2] == 0).all())

    def test_produce_shap_one_hot(self):
        e_transforms = self.one_hot_encoder
        lfc = LocalFeatureContribution(model=self.model_one_hot_filename,
                                       x_orig=self.X_train, e_algorithm='shap',
                                       fit_on_init=True, e_transforms=e_transforms,
                                       interpretable_features=False)
        shap = ShapFeatureContribution(model=self.model_one_hot_filename,
                                       x_orig=self.X_train, fit_on_init=True,
                                       e_transforms=e_transforms)
        self.helper_produce_shap_one_hot(lfc)
        self.helper_produce_shap_one_hot(shap)

    def helper_produce_shap_one_hot(self, explainer):
        x_one_dim = pd.DataFrame([[2, 10, 10]], columns=["A", "B", "C"])
        x_multi_dim = pd.DataFrame([[4, 1, 1],
                                    [6, 2, 3]], columns=["A", "B", "C"])
        contributions = explainer.produce(x_one_dim)
        self.assertEqual(x_one_dim.shape, contributions.shape)
        self.assertAlmostEqual(contributions["A"][0], -1, places=5)
        self.assertAlmostEqual(contributions["B"][0], 0, places=5)
        self.assertAlmostEqual(contributions["C"][0], 0, places=5)

        contributions = explainer.produce(x_multi_dim)
        self.assertEqual(x_multi_dim.shape, contributions.shape)
        self.assertAlmostEqual(contributions["A"][0], 0, places=5)
        self.assertAlmostEqual(contributions["A"][1], 1, places=5)
        self.assertTrue((contributions["B"] == 0).all())
        self.assertTrue((contributions["C"] == 0).all())

    def test_produce_with_renames(self):
        e_transforms = self.one_hot_encoder
        feature_descriptions = {"A": "Feature A", "B": "Feature B"}
        lfc = LocalFeatureContribution(model=self.model_one_hot_filename,
                                       x_orig=self.X_train, e_algorithm='shap',
                                       fit_on_init=True, e_transforms=e_transforms,
                                       interpretable_features=True,
                                       feature_descriptions=feature_descriptions)
        x_one_dim = pd.DataFrame([[2, 10, 10]], columns=["A", "B", "C"])

        contributions = lfc.produce(x_one_dim)
        self.assertEqual(x_one_dim.shape, contributions.shape)
        self.assertAlmostEqual(contributions["Feature A"][0], -1, places=5)
        self.assertAlmostEqual(contributions["Feature B"][0], 0, places=5)
        self.assertAlmostEqual(contributions["C"][0], 0, places=5)
