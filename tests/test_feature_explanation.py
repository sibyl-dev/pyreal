import numpy as np
from sklearn.linear_model import Lasso
import pandas as pd

"""Tests for `explanation_toolkit` package."""

import unittest

from explanation_toolkit import feature_explanation


def identity(x):
    return x


class TestFeatureExplanation(unittest.TestCase):
    """Tests for `explanation_toolkit` package."""

    def setUp(self):
        """Set up test fixtures"""
        self.conversions = [identity, np.array, pd.DataFrame]

        self.fc = feature_explanation.FeatureContributions()
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
        for conv in self.conversions:
            self.X_train = conv(self.X_train)
            output = self.fc.fit_contributions(self.model, self.X_train)
            self.assertIsNone(output)
            self.assertIsNotNone(self.fc.explainer)

    def test_get_contributions(self):
        x_low = np.array([1,5,6])
        x_high = [6,3,2]

        self.assertIsNone(self.fc.explainer)
        with self.assertRaises(AssertionError):
            self.fc.get_contributions(x_low)

        for conv in self.conversions:
            self.X_train = conv(self.X_train)
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



