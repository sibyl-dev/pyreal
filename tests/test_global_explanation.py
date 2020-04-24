#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso

"""Tests for `explanation_toolkit` package."""

import unittest

from explanation_toolkit import global_explanation

# TODO Fix some of the hardcoding in these tests, it'll be a problem later
# TODO Maybe remove run_type_test_suite and manually iterate conversions2d in each
#      test to maximize flexibility


def predict_test(X):
    return X[:,0]


def run_type_test_suite(func, inputs):
    func(*inputs)
    for i in range(len(inputs)):
        inputs[i] = np.array(inputs[i])
    func(*inputs)
    for i in range(len(inputs)):
        inputs[i] = pd.DataFrame(inputs[i])
    func(*inputs)


def identity(X):
    return X


class TestGlobalExplanation(unittest.TestCase):
    """Tests for `explanation_toolkit` package."""

    def setUp(self):
        self.conversions2d = [identity, np.array, pd.DataFrame]
        self.conversions1d = [identity, np.array, pd.Series]
        pass

    def tearDown(self):
        """Tear down test fixtures, if any."""
        pass

    def test_get_global_importance(self):
        # Set up tests
        for conv2d, conv1d in [
                (conv2d, conv1d) for conv2d in self.conversions2d
                                 for conv1d in self.conversions1d]:
            self.helper_global_importance(conv2d, conv1d)

    def test_get_rows_by_output(self):
        for conv in self.conversions2d:
            self.helper_get_rows_by_output(conv)

    def test_summary_categorical(self):
        for conv in self.conversions2d:
            self.helper_summary_categorical(conv)

    def test_summary_numeric(self):
        for conv in self.conversions2d:
            self.helper_summary_numeric(conv)

    def helper_global_importance(self, conv2d, conv1d):
        X = conv2d([[3, 4, 2],
                  [5, 3, 6],
                  [0, 1, 2]])
        y = conv1d([2, 4, 1])
        weights = [1, 0, 0]
        model = Lasso()
        model.fit(X, y)
        model.coef_ = np.array(weights)

        importances = global_explanation.get_global_importance(
            model, X, y)
        self.assertTrue(len(importances) == 3)
        self.assertTrue(importances[0] > 0.01)
        self.assertAlmostEqual(importances[1], 0, 4)
        self.assertAlmostEqual(importances[2], 0, 4)

    def helper_get_rows_by_output(self, conv):
        X = conv([[0, 4, 2],
                  [1, 3, 6],
                  [0, 1, 2]])
        row_labels = ["p", "q", "r"]
        rows_0 = [0, 2]
        labels_0 = ["p", "r"]
        rows_1 = [1]
        labels_1 = ["q"]
        output_0 = global_explanation.get_rows_by_output(0, predict_test, X)
        output_1 = global_explanation.get_rows_by_output(1, predict_test, X)
        self.assertTrue(np.array_equal(rows_0, output_0))
        self.assertTrue(np.array_equal(rows_1, output_1))

        output_0 = global_explanation.get_rows_by_output(0, predict_test, X,
                                                         row_labels=row_labels)
        output_1 = global_explanation.get_rows_by_output(1, predict_test, X,
                                                         row_labels=row_labels)
        self.assertTrue(np.array_equal(labels_0, output_0))
        self.assertTrue(np.array_equal(labels_1, output_1))

    def helper_summary_categorical(self, conv):
        X = conv([[0, 0, 0],
                  [1, 0, 0],
                  [2, 0, 1]])
        correct_values = [[0, 1, 2], [0], [0, 1]]
        correct_counts = [[1, 1, 1], [3], [2, 1]]
        result_values, result_counts = global_explanation.summary_categorical(X)
        self.assertEqual(len(result_values), 3)
        self.assertEqual(len(result_counts), 3)
        for i in range(len(X[0])):
            self.assertTrue(np.array_equal(correct_values[i], result_values[i]))
            self.assertTrue(np.array_equal(correct_counts[i], result_counts[i]))
            self.assertEqual(len(result_values[i]), len(result_counts[i]))

    def helper_summary_numeric(self, conv):
        X = conv([[0, 0, 0],
                  [1, 2, 0],
                  [2, 8, 0],
                  [3, 6, 0],
                  [4, 4, 0]])
        correct = [[0, 1, 2, 3, 4],
                   [0, 2, 4, 6, 8],
                   [0, 0, 0, 0, 0]]
        result = global_explanation.summary_numeric(X)
        self.assertEqual(len(result), 3)
        for i in range(3):
            self.assertTrue(np.array_equal(result[i], correct[i]))
            self.assertEqual(len(result[i]), 5)
