#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso

from sibyl import global_explanation


class TestEstimator:
    def fit(self, X, y):
        return self

    def predict(self, X):
        return X.iloc[:, 0]


def predict_test(X):
    return X.iloc[:, 0]


class TestGlobalExplanation(unittest.TestCase):
    """Tests for `sibyl` package."""

    def setUp(self):
        pass

    def tearDown(self):
        """Tear down test fixtures, if any."""
        pass

    def test_get_global_importance(self):
        X = pd.DataFrame([[3, 4, 2],
                          [5, 3, 6],
                          [0, 1, 2]], columns=["A", "B", "C"])
        y = [3, 5, 0]
        weights = [1, 0, 0]
        model = Lasso()
        model.fit(X, y)
        model.coef_ = np.array(weights)
        importances = global_explanation.get_global_importance(
            model, X, y)
        self.assertTrue(len(importances) == 3)
        self.assertTrue(float(importances.loc["A"]) > 0.01)
        self.assertAlmostEqual(float(importances.loc["B"]), 0, 4)
        self.assertAlmostEqual(float(importances.loc["C"]), 0, 4)

    def test_consolidate_importances(self):
        importances = [0, 0, 1, 3, 6, 3]
        categories = [[0, 1], [3, 4, 5], [5]]
        max_importances = [0, 6, 3]
        mean_importances = [0, 4, 3]

        max_results = global_explanation.consolidate_importances(
            importances, categories, algorithm="max")
        self.assertTrue(np.array_equal(max_importances, max_results))

        mean_results = global_explanation.consolidate_importances(
            importances, categories, algorithm="mean")
        self.assertTrue(np.array_equal(mean_importances, mean_results))

        with self.assertRaises(ValueError):
            global_explanation.consolidate_importances(
                importances, categories, algorithm="fake_algorithm")

    def test_get_rows_by_output(self):
        X = pd.DataFrame([[0, 4, 2],
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

    def test_summary_categorical(self):
        X = pd.DataFrame([[0, 0, "0"],
                          [1, 0, "0"],
                          [2, 0, "1"]])
        correct_values = [[0, 1, 2], [0], ["0", "1"]]
        correct_counts = [[1, 1, 1], [3], [2, 1]]
        result_values, result_counts = global_explanation.summary_categorical(X)
        print("result_values: ", result_values)
        print("result_counts: ", result_counts)
        self.assertEqual(len(result_values), 3)
        self.assertEqual(len(result_counts), 3)
        for i in range(3):
            self.assertTrue(np.array_equal(correct_values[i], result_values[i]))
            self.assertTrue(np.array_equal(correct_counts[i], result_counts[i]))
            self.assertEqual(len(result_values[i]), len(result_counts[i]))

    def test_summary_numeric(self):
        X = pd.DataFrame([[0, 0, 0],
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

    def test_overview_categorical(self):
        X = pd.DataFrame([[0, 0, 0],
                          [9, 9, 9],
                          [0, 0, 2],
                          [1, 0, 1]])
        correct_values_0 = [[0], [0], [0, 2]]
        correct_counts_0 = [[2], [2], [1, 1]]

        result_values_0, result_counts_0 = \
            global_explanation.overview_categorical(0, predict_test, X)
        self.assertEqual(len(result_values_0), 3)
        self.assertEqual(len(result_counts_0), 3)
        for i in range(3):
            self.assertTrue(
                np.array_equal(correct_values_0[i], result_values_0[i]))
            self.assertTrue(
                np.array_equal(correct_counts_0[i], result_counts_0[i]))
            self.assertEqual(len(result_values_0[i]), len(result_counts_0[i]))

        correct_values_1 = [[1], [1]]
        correct_counts_1 = [[1], [1]]

        result_values_1, result_counts_1 = \
            global_explanation.overview_categorical(1, predict_test, X,
                                                    features=[0, 2])
        self.assertEqual(len(result_values_1), 2)
        self.assertEqual(len(result_counts_1), 2)
        for i in range(2):
            self.assertTrue(
                np.array_equal(correct_values_1[i], result_values_1[i]))
            self.assertTrue(
                np.array_equal(correct_counts_1[i], result_counts_1[i]))
            self.assertEqual(len(result_values_1[i]), len(result_counts_1[i]))

    def test_overview_numeric(self):
        X = pd.DataFrame([[0, 0, 0, 0],
                          [0, 1, 2, 0],
                          [0, 2, 8, 0],
                          [0, 3, 6, 0],
                          [0, 4, 4, 0],
                          [1, 1, 1, 1]])
        correct_0 = [[0, 1, 2, 3, 4],
                     [0, 2, 4, 6, 8],
                     [0, 0, 0, 0, 0]]

        result_0 = global_explanation.overview_numeric(
            0, predict_test, X, features=[1, 2, 3])
        self.assertEqual(len(result_0), 3)
        for i in range(3):
            self.assertTrue(np.array_equal(correct_0[i], result_0[i]))
            self.assertEqual(len(result_0[i]), 5)
