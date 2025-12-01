from __future__ import annotations

import unittest
import numpy as np
import pandas as pd

from cluster_maker.preprocessing import select_features, standardise_features

class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        #Creating test data
        self.df_valid = pd.DataFrame({
            'x': [1.0, 2.0, 3.0, 4.0, 5.0],
            'y': [10.0, 20.0, 30.0, 40.0, 50.0],
            'label': ['A', 'B', 'C', 'D', 'E'],
        })


# Test 1: Verifying for missing columns in the data
    """ A problem could be if the user requests features that don't exist in the DataFrame.
    This is a common mistake (typo in column name or wrong dataset).
    Without validation, the code could fail later with a less clear error.
    This test ensures missing columns are caught with a clear error.
    """
    def test_select_features_missing_columns_raises_keyerror(self):
        with self.assertRaises(KeyError) as context:
            select_features(self.df_valid, ['x', 'nonexistent_column'])
        
        #Error message identifies the missing column
        error_msg = str(context.exception)
        self.assertIn("nonexistent_column", error_msg)
        self.assertIn("missing", error_msg.lower())


# Test 2: Verifying for non-numeric features causing data type mismatch
    """
    Another problem could be if the user tries to use non-numeric columns (strings, dates, objects) for clustering.
    K-means requires Euclidean distance, which is undefined for non-numeric data.
    Without validation, clustering either crashes or produces meaningless results.
    This test ensures non-numeric columns are rejected before reaching the clustering algorithm.
    """
    def test_select_features_non_numeric_columns_raises_typeerror(self):
        with self.assertRaises(TypeError) as context:
            select_features(self.df_valid, ['x', 'label'])
        
        #Error message identifies the non-numeric column
        error_msg = str(context.exception).lower()
        self.assertIn("label", error_msg.lower())
        self.assertIn("not numeric", error_msg)


# Test 3: Verifying for standardisation so large-ranged features don't dominate
        """
        Another issue would be that features with different scales can dominate distance metrics.
        Without standardisation, too big or too small features skew results, making other features irrelevant.
        This test ensures standardisation produces zero mean and unit variance.
        """
    def test_standardise_features_produces_zero_mean_unit_variance(self):

        # Create data with deliberately different scales
        X = np.array([
            [1.0, 100.0],     # Small x, large y
            [2.0, 200.0],
            [3.0, 300.0],
            [4.0, 400.0],
        ], dtype=float)
        
        X_scaled = standardise_features(X)
        
        # Check shape is preserved
        self.assertEqual(X_scaled.shape, X.shape)
        
        # Check each column has approximately zero mean
        means = X_scaled.mean(axis=0)
        np.testing.assert_array_almost_equal(means, [0.0, 0.0], decimal=10)
        
        # Check each column has approximately unit variance (std = 1.0)
        stds = X_scaled.std(axis=0, ddof=0)
        np.testing.assert_array_almost_equal(stds, [1.0, 1.0], decimal=10)


if __name__ == "__main__":
    unittest.main()