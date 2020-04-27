#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `explanation_toolkit` package."""

import unittest
import numpy as np
import pandas as pd

from explanation_toolkit import nearest_neighbor_explanation
from utils.distance import PartialFeatureDistance


class TestNearestNeighborExplanation(unittest.TestCase):
    """Tests for `explanation_toolkit` package."""

    def setUp(self):
        self.ee = nearest_neighbor_explanation.NearestNeighborExplanation()
        self.X = [[0, 0, 0],
                  [1, 1, 1],
                  [2, 3, 2],
                  [10, 11, 12]]
        self.y = [1, 1, 0, 0]

    def tearDown(self):
        pass

    def test_fit_nearest_neighbor(self):
        self.assertIsNone(self.ee.nbrs)
        self.assertIsNone(self.ee.y)

        output = self.ee.fit_nearest_neighbor(self.X)
        self.assertIsNotNone(self.ee.nbrs)
        self.assertIsNone(self.ee.y)
        self.assertIsNone(output)

        output = self.ee.fit_nearest_neighbor(np.array(self.X), self.y)
        self.assertIsNotNone(self.ee.nbrs)
        self.assertIsNotNone(self.ee.y)
        self.assertIsNone(output)

        self.ee.fit_nearest_neighbor(pd.DataFrame(self.X), np.array(self.y))
        self.ee.fit_nearest_neighbor(pd.DataFrame(self.X), pd.Series(self.y))

    def test_nearest_neighbor_no_provided_y(self):
        x = [[.3, .3, .3]]

        with self.assertRaises(AssertionError):
            self.ee.nearest_neighbor(self.X)

        self.ee.fit_nearest_neighbor(self.X)
        with self.assertRaises(ValueError):
            self.ee.nearest_neighbor(self.X, desired_y=1)

        output1 = self.ee.nearest_neighbor(x)
        self.assertTrue(np.array_equal(output1, [0]))

        output2 = self.ee.nearest_neighbor(x, N=2)
        self.assertTrue(np.array_equal(output2, [0, 1]))

        output3 = self.ee.nearest_neighbor(x, N=1, desired_y=0, y=self.y)
        self.assertTrue(np.array_equal(output3, [2]))

        output4 = self.ee.nearest_neighbor(x, N=3, desired_y=0, y=self.y)
        self.assertTrue(np.array_equal(output4, [2, 3]))

        output5 = self.ee.nearest_neighbor(x, N=3, desired_y=0, y=self.y,
                                           search_by=1, search_depth=1)
        self.assertTrue(np.array_equal(output5, []))

    def test_nearest_neighbor_provided_y(self):
        x = [[.3, .3, .3]]

        with self.assertRaises(AssertionError):
            self.ee.nearest_neighbor(self.X)

        self.ee.fit_nearest_neighbor(self.X, self.y)

        output3 = self.ee.nearest_neighbor(x, N=1, desired_y=0)
        self.assertTrue(np.array_equal(output3, [2]))

        output4 = self.ee.nearest_neighbor(x, N=3, desired_y=0)
        self.assertTrue(np.array_equal(output4, [2, 3]))

        output5 = self.ee.nearest_neighbor(x, N=3, desired_y=0,
                                           search_by=1, search_depth=1)
        self.assertTrue(np.array_equal(output5, []))

    def test_nearest_neighbor_custom_metric(self):
        x = [[7, 0, 0]]
        self.ee.fit_nearest_neighbor(self.X)

        output1 = self.ee.nearest_neighbor(x)
        self.assertTrue(np.array_equal(output1, [1]))

        dist = PartialFeatureDistance([0]).distance
        self.ee.fit_nearest_neighbor(self.X, metric=dist)

        output2 = self.ee.nearest_neighbor(x)
        self.assertTrue(np.array_equal(output2, [3]))
