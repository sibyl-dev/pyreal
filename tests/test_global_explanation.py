#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso

"""Tests for `explanation_toolkit` package."""

import unittest

from explanation_toolkit import global_explanation


def predict_test(X):
    return X.iloc[:,0]


class TestGlobalExplanation(unittest.TestCase):
    """Tests for `explanation_toolkit` package."""

    def setUp(self):
        """Set up test fixtures, if any."""
        pass

    def tearDown(self):
        """Tear down test fixtures, if any."""
        pass

    def test_get_global_importance(self):
        weights = [1,0,0]
        X_train = [[3,4,2],
                   [5,3,6],
                   [0,1,2]]
        y_train = [2, 4, 1]
        model = Lasso()
        model.fit(X_train, y_train)
        model.coef_ = np.array(weights)

        importances = global_explanation.get_global_importance(
            model, X_train, y_train)
        self.assertTrue(len(importances) == 3)
        self.assertTrue(importances[0] > 0.01)
        self.assertAlmostEqual(importances[1], 0, 4)
        self.assertAlmostEqual(importances[2], 0, 4)

    def test_get_xs_with_predictions(self):
        X = pd.DataFrame([[0, 4, 2],
                          [1, 3, 6],
                          [0, 1, 2]])
        X_0 = pd.DataFrame([[0, 4, 2],
                            [0, 1, 2]])
        X_1 = pd.DataFrame([[1, 3, 6]])

        output_0 = global_explanation.get_xs_with_predictions(0, predict_test, X)
        output_1 = global_explanation.get_xs_with_predictions(1, predict_test, X)
        self.assertTrue(np.array_equal(X_0, output_0))
        self.assertTrue(np.array_equal(X_1, output_1))

    def test_summary_count(self):
        X_np = [[0, 0, 0],
                [1, 0, 0],
                [2, 0, 1]]
        X = pd.DataFrame(X_np, columns=["A", "B", "C"])
        correct = {"A":np.array([[0,1,2],[1,1,1]]),
                   "B":np.array([[0],[3]]),
                   "C":np.array([[0,1],[2,1]])}
        result = global_explanation.summary_count(X)
        self.check_summary_count_output(result, correct)

    def test_summary_metrics(self):
        X_np = [[0, 0, 0],
                [1, 2, 0],
                [2, 8, 0],
                [3, 6, 0],
                [4, 4, 0]]
        X = pd.DataFrame(X_np, columns=["A", "B", "C"])
        correct = {"A":np.array([0,1,2,3,4]),
                   "B":np.array([0,2,4,6,8]),
                   "C":np.array([0,0,0,0,0])}
        result = global_explanation.summary_metrics(X)
        self.check_summary_metrics_output(result, correct)

    def test_overview_categorical(self):
        X = pd.DataFrame([[0, 4, 2],
                          [1, 3, 6],
                          [0, 1, 2]], columns=["A", "B", "C"])
        correct_0 = {"A":np.array([[0],[2]]),
                     "B":np.array([[4,1],[1,1]]),
                     "C":np.array([[2],[2]])}
        correct_1 = {"A":np.array([[1],[1]]),
                     "B":np.array([[3],[1]]),
                     "C":np.array([[6],[1]])}
        result_0 = global_explanation.overview_categorical(0, predict_test, X)
        result_1 = global_explanation.overview_categorical(1, predict_test, X)
        self.check_summary_count_output(result_0, correct_0)
        self.check_summary_count_output(correct_1, result_1)

    def test_overview_metrics(self):
        X = pd.DataFrame([[0, 4, 2],
                          [1, 3, 6],
                          [0, 1, 2]], columns=["A", "B", "C"])
        correct_0 = {"A":np.array([0,0,0,0,0]),
                     "B":np.array([1,1.75,2.5,3.25,4]),
                     "C":np.array([2,2,2,2,2])}
        correct_1 = {"A":np.array([1,1.0,1.0,1.0,1]),
                     "B":np.array([3,3.0,3.0,3.0,3]),
                     "C":np.array([6,6.0,6.0,6.0,6])}
        result_0 = global_explanation.overview_metrics(0, predict_test, X)
        result_1 = global_explanation.overview_metrics(1, predict_test, X)
        self.check_summary_metrics_output(result_0, correct_0)
        self.check_summary_metrics_output(correct_1, result_1)

    def check_summary_count_output(self, result, correct):
        self.assertEqual(result.keys(), correct.keys())
        for feature in correct:
            correct_unique, correct_count = correct[feature]
            result_unique, result_count = result[feature]
            correct_sort = np.argsort(correct_unique)
            result_sort = np.argsort(result_unique)

            self.assertTrue(np.array_equal(correct_unique[correct_sort],
                                           result_unique[result_sort]))
            self.assertTrue(np.array_equal(correct_count[correct_sort],
                                           result_count[result_sort]))

    def check_summary_metrics_output(self, result, correct):
        self.assertEqual(result.keys(), correct.keys())
        for feature in correct:
            correct_metrics = correct[feature]
            result_metrics = result[feature]
            self.assertTrue(np.array_equal(correct_metrics,
                                           result_metrics))
