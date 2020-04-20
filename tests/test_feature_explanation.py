import numpy as np
from sklearn.linear_model import Lasso
import pandas as pd

"""Tests for `explanation_toolkit` package."""

import unittest

from explanation_toolkit import feature_explanation


def predict_test(X):
    return X.iloc[:,0]


class TestFeatureExplanation(unittest.TestCase):
    """Tests for `explanation_toolkit` package."""
    X_train = None

    def setUp(self):
        """Set up test fixtures, if any."""
        pass

    def tearDown(self):
        """Tear down test fixtures, if any."""
        pass

    def test_fit_contributions(self):
        X_train = pd.DataFrame([[1,1,1],
                                [1,3,4],
                                [6,7,2]])
        output = feature_explanation.fit_contributions(model, X_train)
        self.assertTrue(True)

    def test_get_contributions(self):
        self.assertTrue(True)
