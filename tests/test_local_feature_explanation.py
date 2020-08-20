import os
import unittest

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso

from sibyl import local_feature_explanation


def identity(x):
    return x


class TestFeatureExplanation(unittest.TestCase):
    """Tests for `sibyl` package."""

    def setUp(self):
        """Set up test fixtures"""
        self.conversions2d = [identity, np.array, pd.DataFrame]
        self.conversions1d = [identity, np.array, pd.Series]

        self.X_train = pd.DataFrame([[1, 1, 1],
                                     [4, 3, 4],
                                     [6, 7, 2]])
        self.y_train = pd.Series([1, 4, 6])
        self.model = Lasso()
        self.model.fit(self.X_train, self.y_train)
        self.weights = [1, 0, 0]
        self.model.coef_ = np.array(self.weights)

    def tearDown(self):
        """Tear down test fixtures, if any."""
        pass

    def test_fit_contribution_explainer(self):
        filepath = "temp"
        with open(filepath, "w+b") as savefile:
            output = local_feature_explanation.fit_contribution_explainer(
                self.model, self.X_train,
                savefile=savefile, return_result=True)
            self.assertIsNotNone(output)
            assert os.path.exists(savefile.name)
        assert (os.path.getsize(savefile.name) > 0)
        os.remove(savefile.name)

    def test_load_contribution_explainer(self):
        filename = "temp"
        with open(filename, "w+b") as savefile:
            local_feature_explanation.fit_contribution_explainer(
                self.model, self.X_train, savefile=savefile)
        with open(filename, "rb") as savefile:
            explainer = local_feature_explanation.load_contribution_explainer(
                savefile)
        self.assertIsNotNone(explainer)
        os.remove(filename)

    def test_get_contributions(self):
        x_low = pd.DataFrame([[1, 5, 6]])
        x_high = pd.DataFrame([[6, 3, 2]])

        explainer = local_feature_explanation.fit_contribution_explainer(
            self.model, self.X_train, return_result=True)
        output = local_feature_explanation.get_contributions(x_low, explainer)

        self.assertTrue(output.shape == (1, 3))
        self.assertTrue(float(output[0]) < -0.01)
        self.assertAlmostEqual(float(output[1]), 0, 4)
        self.assertAlmostEqual(float(output[2]), 0, 4)

        output = local_feature_explanation.get_contributions(x_high, explainer)

        self.assertTrue(output.shape == (1, 3))
        self.assertTrue(float(output[0]) > 0.01)
        self.assertAlmostEqual(float(output[1]), 0, 4)
        self.assertAlmostEqual(float(output[2]), 0, 4)
