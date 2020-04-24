#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `explanation_toolkit` package."""

import unittest
import numpy as np
import pandas as pd

from explanation_toolkit import nearest_neighbor_explanation


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

    def test_nearest_neighbor(self):
        """Test something."""
        self.assertTrue(True)
