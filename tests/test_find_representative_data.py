import unittest
import pandas as pd
import numpy as np

from utils import find_representative_data


class TestFeatureExplanation(unittest.TestCase):
    """Tests for `explanation_toolkit` package."""

    def setUp(self):
        self.X_data = pd.DataFrame([[0, 0], [1, 0], [1, 1], [1.3, 0],
                           [4, 4], [5, 3], [3.4, 3.3], [4, 5],
                           [10, 11], [11, 11], [10.5, 10.5], [14, 14]])
        self.three_medoids = [4,1,9]
        # TODO: make this less order dependent
        self.three_medoids_labels = [1,1,1,1,0,0,0,0,2,2,2,2]

    def test_kmedoids(self):
        results = find_representative_data.kmedoids(self.X_data, 3, 2)
        medoids = results[0]
        labels = results[1]
        print(labels)

        self.assertTrue(np.array_equal(medoids, self.three_medoids))
        self.assertTrue(np.array_equal(labels, self.three_medoids_labels))

