#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `sibyl` package."""

import unittest
import numpy as np
import pandas as pd
import os

from real.explainers import nearest_neighbor_explanation
from utils.distance import PartialFeatureDistance


class TestNearestNeighborExplanation(unittest.TestCase):
    """Tests for `sibyl` package."""

    def setUp(self):
        self.X = [[0, 0, 0],
                  [1, 1, 1],
                  [2, 3, 2],
                  [10, 11, 12]]
        self.y = [1, 1, 0, 0]

    def tearDown(self):
        pass

    def test_fit_nearest_neighbor(self):
        output = nearest_neighbor_explanation.fit_nearest_neighbor(
            self.X, return_result=True)
        self.assertIsNotNone(output)

        filepath = "temp"
        with open(filepath, "w+b") as savefile:
            output = nearest_neighbor_explanation.fit_nearest_neighbor(
                self.X, savefile=savefile)
            self.assertIsNone(output)

        assert (os.path.getsize(savefile.name) > 0)
        os.remove(savefile.name)

        nearest_neighbor_explanation.fit_nearest_neighbor(pd.DataFrame(self.X))

    def test_load_nearest_neighbor(self):
        filename = "temp"
        with open(filename, "w+b") as savefile:
            nearest_neighbor_explanation.fit_nearest_neighbor(
                self.X, savefile=savefile)
        with open(filename, "rb") as savefile:
            nbrs = nearest_neighbor_explanation.load_nearest_neighbor(savefile)
        self.assertIsNotNone(nbrs)
        os.remove(filename)

    def test_nearest_neighbor_no_provided_y(self):
        x = [[.3, .3, .3]]
        nbrs = nearest_neighbor_explanation.fit_nearest_neighbor(
            self.X, return_result=True)

        with self.assertRaises(ValueError):
            nearest_neighbor_explanation.nearest_neighbor(nbrs, self.X, desired_y=1)

        output1 = nearest_neighbor_explanation.nearest_neighbor(nbrs, x)
        self.assertTrue(np.array_equal(output1, [0]))

        output2 = nearest_neighbor_explanation.nearest_neighbor(nbrs, x, N=2)
        self.assertTrue(np.array_equal(output2, [0, 1]))

        output3 = nearest_neighbor_explanation.nearest_neighbor(
            nbrs, x, N=1, desired_y=0, y=self.y, search_by=1)
        self.assertTrue(np.array_equal(output3, [2]))

        output4 = nearest_neighbor_explanation.nearest_neighbor(
            nbrs, x, N=3, desired_y=0, y=self.y, search_by=1)
        self.assertTrue(np.array_equal(output4, [2, 3]))

        output5 = nearest_neighbor_explanation.nearest_neighbor(
            nbrs, x, N=3, desired_y=0, y=self.y,search_by=1, search_depth=1)
        self.assertTrue(np.array_equal(output5, []))

    def test_nearest_neighbor_custom_metric(self):
        x = [[7, 0, 0]]
        nbrs = nearest_neighbor_explanation.fit_nearest_neighbor(
            self.X, return_result=True)

        output1 = nearest_neighbor_explanation.nearest_neighbor(nbrs, x)
        self.assertTrue(np.array_equal(output1, [1]))

        dist = PartialFeatureDistance([0]).distance
        nbrs = nearest_neighbor_explanation.fit_nearest_neighbor(
            self.X, metric=dist, return_result=True)

        output2 = nearest_neighbor_explanation.nearest_neighbor(nbrs, x)
        self.assertTrue(np.array_equal(output2, [3]))
