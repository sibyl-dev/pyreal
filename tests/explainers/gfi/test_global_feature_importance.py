import os
import pickle
import unittest

import numpy as np
import pandas as pd
from shap import LinearExplainer
from sklearn.linear_model import LinearRegression

from pyreal.explainers import GlobalFeatureImportance, ShapFeatureImportance
from pyreal.utils.transformer import OneHotEncoderWrapper

TEST_ROOT = os.path.dirname(os.path.abspath(__file__))


def identity(x):
    return x


class TestGlobalFeatureImportance(unittest.TestCase):
    """Tests for `pyreal` package."""

    def setUp(self):
        """Set up test fixtures"""
        try:
            os.makedirs(os.path.join(TEST_ROOT, "../../data"))
        except FileExistsError:
            pass

        self.X_train = pd.DataFrame([[2, 1, 3, 3],
                                     [4, 3, 4, 3],
                                     [6, 7, 2, 3]], columns=["A", "B", "C", "D"])
        self.y_train = pd.DataFrame([2, 4, 6])
        self.expected = np.mean(self.y_train)[0]
        model_no_transforms = LinearRegression()
        model_no_transforms.fit(self.X_train, self.y_train)
        model_no_transforms.coef_ = np.array([1, 0, 0, 0])
        self.model_no_transforms_filename = os.path.join(TEST_ROOT, "../../data",
                                                         "model_no_transforms.pkl")
        with open(self.model_no_transforms_filename, "wb") as f:
            pickle.dump(model_no_transforms, f)

        self.one_hot_encoder = OneHotEncoderWrapper(feature_list=["A"])
        self.one_hot_encoder.fit(self.X_train)
        self.X_transformed = self.one_hot_encoder.transform(self.X_train)
        self.y_transformed = pd.DataFrame([1, 2, 3])
        self.expected_transformed = np.mean(self.y_transformed)[0]
        model_one_hot = LinearRegression()
        model_one_hot.fit(self.X_transformed, self.y_transformed)
        model_one_hot.coef_ = np.array([0, 0, 0, 1, 2, 3])
        self.model_one_hot_filename = os.path.join(TEST_ROOT, "../../data", "model_one_hot.pkl")
        with open(self.model_one_hot_filename, "wb") as f:
            pickle.dump(model_one_hot, f)

    def tearDown(self):
        """Tear down test fixtures"""
        os.remove(self.model_no_transforms_filename)
        os.remove(self.model_one_hot_filename)

    def test_fit_shap_no_transforms(self):
        gfi_object = GlobalFeatureImportance(
            model=self.model_no_transforms_filename,
            x_train_orig=self.X_train, e_algorithm='shap')
        gfi_object.fit()

        shap = ShapFeatureImportance(
            model=self.model_no_transforms_filename, x_train_orig=self.X_train)
        shap.fit()
        self.assertIsNotNone(shap.explainer)
        self.assertIsInstance(shap.explainer, LinearExplainer)

    def test_produce_shap_no_transforms(self):
        gfi = GlobalFeatureImportance(model=self.model_no_transforms_filename,
                                      x_train_orig=self.X_train, e_algorithm='shap',
                                      fit_on_init=True)
        shap = ShapFeatureImportance(
            model=self.model_no_transforms_filename, x_train_orig=self.X_train,
            fit_on_init=True)

        self.helper_produce_shap_no_transforms(gfi)
        self.helper_produce_shap_no_transforms(shap)

    def helper_produce_shap_no_transforms(self, explainer):
        importances = explainer.produce()
        self.assertEqual(importances.shape, (1, self.X_train.shape[1]))
        self.assertAlmostEqual(importances["A"][0], (4 / 3))
        self.assertEqual(importances["B"][0], 0)
        self.assertEqual(importances["C"][0], 0)
        self.assertEqual(importances["D"][0], 0)

    def test_produce_shap_one_hot(self):
        e_transforms = self.one_hot_encoder
        gfi = GlobalFeatureImportance(model=self.model_one_hot_filename,
                                      x_train_orig=self.X_train, e_algorithm='shap',
                                      fit_on_init=True, e_transforms=e_transforms,
                                      interpretable_features=False)
        shap = ShapFeatureImportance(model=self.model_one_hot_filename,
                                     x_train_orig=self.X_train, fit_on_init=True,
                                     e_transforms=e_transforms)
        self.helper_produce_shap_one_hot(gfi)
        self.helper_produce_shap_one_hot(shap)

    def helper_produce_shap_one_hot(self, explainer):
        importances = explainer.produce()
        self.assertEqual(importances.shape, (1, self.X_train.shape[1]))
        self.assertAlmostEqual(importances["A"][0], (8 / 3))
        self.assertEqual(importances["B"][0], 0)
        self.assertEqual(importances["C"][0], 0)
        self.assertEqual(importances["D"][0], 0)

    def test_produce_with_renames(self):
        e_transforms = self.one_hot_encoder
        feature_descriptions = {"A": "Feature A", "B": "Feature B"}
        gfi = GlobalFeatureImportance(model=self.model_one_hot_filename,
                                      x_train_orig=self.X_train, e_algorithm='shap',
                                      fit_on_init=True, e_transforms=e_transforms,
                                      interpretable_features=True,
                                      feature_descriptions=feature_descriptions)

        importances = gfi.produce()
        self.assertEqual(importances.shape, (1, self.X_train.shape[1]))
        self.assertAlmostEqual(importances["Feature A"][0], (8 / 3))
        self.assertEqual(importances["Feature B"][0], 0)
        self.assertEqual(importances["C"][0], 0)
        self.assertEqual(importances["D"][0], 0)
