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


def identity(X):
    return X


class TestCounterfactualExplanation(unittest.TestCase):
    """Tests for `explanation_toolkit` package."""

    def setUp(self):
        """Set up test fixtures"""
        pass

    def tearDown(self):
        """Tear down test fixtures, if any."""
        pass

    def test_modify_and_repredict(self):
        self.helper_modify_and_repredict_2D(identity)
        self.helper_modify_and_repredict_2D(np.array)
        self.helper_modify_and_repredict_2D(pd.DataFrame)

        self.helper_modify_and_repredict_1D(identity)
        self.helper_modify_and_repredict_1D(np.array)
        self.helper_modify_and_repredict_1D(pd.Series)

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

    def test_modify_input(self):
        self.helper_modify_input(identity)
        self.helper_modify_input(np.array)
        self.helper_modify_input(pd.DataFrame)

    def helper_modify_and_repredict_2D(self, conv):
        X = conv([[1, 1, 1],
                  [4, 3, 4],
                  [6, 7, 2]])
        pred = [1, 4, 6]
        pred_2 = [2, 7, 13]

        no_change = counterfactual_explanation.modify_and_repredict(
            predict_test, X, features=[], new_values=[])
        self.assertTrue(np.array_equal(pred, no_change))

        no_change_2 = counterfactual_explanation.modify_and_repredict(
            predict_test_2, X, features=[], new_values=[])
        self.assertTrue(np.array_equal(pred_2, no_change_2))

        change_one = counterfactual_explanation.modify_and_repredict(
            predict_test, X, features=[0], new_values=[[2, 2, 2]])
        prediction_change_one = [2, 2, 2]
        self.assertTrue(np.array_equal(change_one, prediction_change_one))

        change_two = counterfactual_explanation.modify_and_repredict(
            predict_test_2, X, features=[0, 1],
            new_values=[[2, 2, 2], [3, 3, 3]])
        prediction_change_two = [5, 5, 5]
        self.assertTrue(np.array_equal(change_two, prediction_change_two))

    def helper_modify_and_repredict_1D(self, conv):
        x = conv([4, 3, 2])
        pred = [4]
        pred_2 = [7]

        no_change = counterfactual_explanation.modify_and_repredict(
            predict_test, x, features=[], new_values=[])
        self.assertTrue(np.array_equal(pred, no_change))

        no_change_2 = counterfactual_explanation.modify_and_repredict(
            predict_test_2, x, features=[], new_values=[])
        self.assertTrue(np.array_equal(pred_2, no_change_2))

        change_one = counterfactual_explanation.modify_and_repredict(
            predict_test, x, features=[0], new_values=[2])
        prediction_change_one = [2]
        self.assertTrue(np.array_equal(change_one, prediction_change_one))

        change_two = counterfactual_explanation.modify_and_repredict(
            predict_test_2, x, features=[0, 1],
            new_values=[6, 7])
        prediction_change_two = [13]
        self.assertTrue(np.array_equal(change_two, prediction_change_two))

    def helper_modify_input(self, conv):
        X = conv([[1, 1, 1],
                  [4, 3, 4],
                  [6, 7, 2]])
        pred = [1, 4, 6]
        mi = counterfactual_explanation.ModifyInput(X)
        self.assertTrue(np.array_equal(X, mi.get()))
        self.assertTrue(np.array_equal(X, mi.original_X))
        self.assertTrue(np.array_equal(X, mi.modified_X))
        self.assertTrue(np.array_equal(pred, mi.predict(predict_test)))

        change = [[2, 2, 2]]
        X2 = [[2, 1, 1],
              [2, 3, 4],
              [2, 7, 2]]
        mi.modify([0], change)
        self.assertTrue(np.array_equal(X2, mi.get()))
        self.assertTrue(np.array_equal([2, 2, 2], mi.predict(predict_test)))

        change2 = [[2, 2, 2], [3, 3, 3]]
        X3 = [[2, 3, 1],
              [2, 3, 4],
              [2, 3, 2]]
        mi.modify([0, 1], change2)
        self.assertTrue(np.array_equal(X3, mi.get()))
        self.assertTrue(np.array_equal([5, 5, 5], mi.predict(predict_test_2)))

        mi.reset(inds=[1])
        self.assertTrue(np.array_equal(X2, mi.get()))
        self.assertTrue(np.array_equal([3, 5, 9], mi.predict(predict_test_2)))

        mi.reset()
        self.assertTrue(np.array_equal(X, mi.get()))
        self.assertTrue(np.array_equal([2, 7, 13], mi.predict(predict_test_2)))

        mi.modify([0], change)
        self.assertTrue(np.array_equal(X2, mi.get()))
        self.assertTrue(np.array_equal([2, 2, 2], mi.predict(predict_test)))
        self.assertTrue(np.array_equal(X, mi.original_X))



