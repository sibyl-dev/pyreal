import numpy as np
from sklearn.linear_model import Lasso
import pandas as pd

"""Tests for `explanation_toolkit` package."""

import unittest

from explanation_toolkit import local_feature_explanation


def identity(x):
    return x


class TestFeatureExplanation(unittest.TestCase):
    """Tests for `explanation_toolkit` package."""

    def setUp(self):
        """Set up test fixtures"""
        self.conversions2d = [identity, np.array, pd.DataFrame]
        self.conversions1d = [identity, np.array, pd.Series]

        self.fc = local_feature_explanation.LocalFeatureContributions()
        self.lime = local_feature_explanation.LimeExplanation()

        self.X_train = [[1, 1, 1],
                        [4, 3, 4],
                        [6, 7, 2]]
        self.y_train = [1, 4, 6]
        self.model = Lasso()
        self.model.fit(self.X_train, self.y_train)
        self.weights = [1, 0, 0]
        self.model.coef_ = np.array(self.weights)

    def tearDown(self):
        """Tear down test fixtures, if any."""
        pass

    def test_fit_contributions(self):
        self.assertIsNone(self.fc.explainer)
        for conv in self.conversions2d:
            self.helper_fit_contributions(conv)

    def test_get_contributions(self):
        self.assertIsNone(self.fc.explainer)
        with self.assertRaises(AssertionError):
            self.fc.get_contributions([0, 0, 0])

        for conv2d, conv1d in [
            (conv2d, conv1d) for conv2d in self.conversions2d
                             for conv1d in self.conversions1d]:
            self.helper_get_contributions(conv2d, conv1d)

    def test_fit_contribution_lime(self):
        self.assertIsNone(self.lime.explainer)
        for conv in self.conversions2d:
            self.helper_fit_contributions_lime(conv)

    def test_get_contributions_lime(self):
        self.assertIsNone(self.lime.explainer)
        with self.assertRaises(AssertionError):
            self.lime.get_contributions([0, 0, 0], self.model.predict)

        for conv2d, conv1d in [
            (conv2d, conv1d) for conv2d in self.conversions2d
                             for conv1d in self.conversions1d]:
            self.helper_get_contributions_lime(conv2d, conv1d)

    def helper_fit_contributions(self, conv):
        self.X_train = conv(self.X_train)
        output = self.fc.fit_contributions(self.model, self.X_train)
        self.assertIsNone(output)
        self.assertIsNotNone(self.fc.explainer)

    def helper_get_contributions(self, conv2d, conv1d):
        self.X_train = conv2d(self.X_train)
        x_low = conv1d([1, 5, 6])
        x_high = conv1d([6, 3, 2])

        self.fc.fit_contributions(self.model, self.X_train)
        output = self.fc.get_contributions(x_low)

        self.assertTrue(len(output) == 3)
        self.assertTrue(output[0] < -0.01)
        self.assertAlmostEqual(output[1], 0, 4)
        self.assertAlmostEqual(output[2], 0, 4)

        output = self.fc.get_contributions(x_high)

        self.assertTrue(len(output) == 3)
        self.assertTrue(output[0] > 0.01)
        self.assertAlmostEqual(output[1], 0, 4)
        self.assertAlmostEqual(output[2], 0, 4)

    def helper_fit_contributions_lime(self, conv):
        self.X_train = conv(self.X_train)
        output = self.lime.fit_contributions(
            self.X_train, feature_names=["A","B","C"])
        self.assertIsNone(output)
        self.assertIsNotNone(self.lime.explainer)

    def helper_get_contributions_lime(self, conv2d, conv1d):
        self.X_train = conv2d(self.X_train)
        x_low = conv1d([1, 5, 6])
        x_high = conv1d([6, 3, 2])

        self.lime.fit_contributions(self.X_train, feature_names=["A","B","C"])
        output = self.lime.get_contributions(
            x_low, self.model.predict)

        self.assertTrue(len(output) == 3)
        self.assertTrue(output[0] < -0.01)
        self.assertAlmostEqual(output[1], 0, 4)
        self.assertAlmostEqual(output[2], 0, 4)

        output = self.lime.get_contributions(x_high, self.model.predict)

        self.assertTrue(len(output) == 3)
        self.assertTrue(output[0] > 0.01)
        self.assertAlmostEqual(output[1], 0, 4)
        self.assertAlmostEqual(output[2], 0, 4)

