import numpy as np
from sklearn.linear_model import Lasso
import pandas as pd

"""Tests for `explanation_toolkit` package."""

import unittest

from explanation_toolkit import counterfactual_explanation


def predict_test(X):
    return X[:,0]

def predict_test_2(X):
    return X[:,0] + X[:,1]


class TestCounterfactualExplanation(unittest.TestCase):
    """Tests for `explanation_toolkit` package."""

    def setUp(self):
        """Set up test fixtures"""
        pass

    def tearDown(self):
        """Tear down test fixtures, if any."""
        pass

    def test_modify_and_repredict(self):
        X = [[1, 1, 1],
             [4, 3, 4],
             [6, 7, 2]]

        # Test with no changes
        original_prediction = [1, 4, 6]
        no_change = counterfactual_explanation.modify_and_repredict(
            predict_test, X, features=[], new_values=[])
        self.assertTrue(np.array_equal(original_prediction, no_change))

        original_prediction_2 = [2, 7, 13]
        no_change_2 = counterfactual_explanation.modify_and_repredict(
            predict_test_2, X, features=[], new_values=[])
        self.assertTrue(np.array_equal(original_prediction_2, no_change_2))

        # test with numeric feature indices
        self.modify_and_repredict_helper(X)
        self.modify_and_repredict_helper(np.array(X))
        self.modify_and_repredict_helper(pd.DataFrame(X))

    def test_binary_flip_all(self):
        x = [1, 0, 0]

        preds_none, values_none = counterfactual_explanation.binary_flip_all(
            predict_test, x, features=[])
        self.assertEqual(preds_none, [])
        self.assertEqual(values_none, [])

        preds_one, values_one = counterfactual_explanation.binary_flip_all(
            predict_test, x, features=[0])
        self.assertEqual(preds_one, [0])
        self.assertEqual(values_one, [0])

        preds_two, values_two = counterfactual_explanation.binary_flip_all(
            predict_test_2, x, features=[0, 1])
        self.assertEqual(preds_two, [0, 2])
        self.assertEqual(values_two, [0, 1])

        preds_all, values_all = counterfactual_explanation.binary_flip_all(
            predict_test_2, x)
        self.assertEqual(preds_all, [0, 2, 1])
        self.assertEqual(values_all, [0, 1, 1])

    def modify_and_repredict_helper(self, X):
        change_one = counterfactual_explanation.modify_and_repredict(
            predict_test, X, features=[0], new_values=[[2, 2, 2]])
        prediction_change_one = [2, 2, 2]
        self.assertTrue(np.array_equal(change_one, prediction_change_one))

        change_two = counterfactual_explanation.modify_and_repredict(
            predict_test_2, X, features=[0, 1],
            new_values=[[2, 2, 2], [3, 3, 3]])
        prediction_change_two = [5, 5, 5]
        self.assertTrue(np.array_equal(change_two, prediction_change_two))
