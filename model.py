"""
NumPy House Price Regression

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - impute_nan_with_mean
import numpy as np

def impute_nan_with_mean(X):
    """Replace every NaN in X with that column's nan-aware mean (all-NaN cols -> 0).
    Args:
        X: (N, F) array-like of floats, may contain NaN.
    Returns:
        (N, F) float ndarray with no NaNs.
    """
    # 1. Make a float copy so we never mutate the caller's array
    X = np.array(X, dtype=float, copy=True)

    # 2. Boolean mask of where the holes are
    nan_mask = np.isnan(X)

    # 3. Per-column mean ignoring NaNs. For a column that is ALL NaN
    #    there is nothing to average, so compute sum/count by hand
    #    and default those columns to 0.0 (avoids the "mean of empty slice" warning).
    counts = (~nan_mask).sum(axis=0)                 # non-NaN values per column
    sums = np.where(nan_mask, 0.0, X).sum(axis=0)    # sum of non-NaN values
    col_means = np.divide(sums, counts,
                          out=np.zeros(X.shape[1]),
                          where=counts > 0)

    # 4. Fill: np.where picks the column mean wherever the mask is True.
    #    col_means has shape (F,), which broadcasts across the N rows.
    return np.where(nan_mask, col_means, X)

# Step 2 - compute_iqr_bounds
import numpy as np

def compute_iqr_bounds(X, k=1.5):
    X = np.asarray(X, dtype=float)
    q1 = np.percentile(X, 25, axis=0)
    q3 = np.percentile(X, 75, axis=0)
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr

# Step 3 - clip_columns
import numpy as np

def clip_columns(X, lower, upper):
    X = np.asarray(X, dtype=float)
    return np.clip(X, np.asarray(lower, dtype=float), np.asarray(upper, dtype=float))

# Step 4 - make_ratio_feature
import numpy as np

def make_ratio_feature(numerator, denominator, eps=1e-8):
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    return numerator / (denominator + eps)

# Step 5 - append_column
import numpy as np

def append_column(X, col):
    X = np.asarray(X)
    col = np.asarray(col).reshape(-1, 1)
    return np.hstack([X, col])

# Step 6 - one_hot_encode
import numpy as np

def one_hot_encode(labels):
    labels = np.asarray(labels)
    categories, idx = np.unique(labels, return_inverse=True)
    idx = idx.reshape(-1)
    out = np.zeros((labels.shape[0], categories.shape[0]), dtype=float)
    out[np.arange(labels.shape[0]), idx] = 1.0
    return out

# Step 7 - fit_standardizer
import numpy as np

def fit_standardizer(X):
    X = np.asarray(X, dtype=float)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    return mean, std

# Step 8 - apply_standardizer
import numpy as np

def apply_standardizer(X, mean, std):
    X = np.asarray(X, dtype=float)
    return (X - np.asarray(mean, dtype=float)) / np.asarray(std, dtype=float)

# Step 9 - add_bias_column
import numpy as np

def add_bias_column(X):
    X = np.asarray(X, dtype=float)
    return np.hstack([np.ones((X.shape[0], 1)), X])

# Step 10 - make_shuffled_indices (not yet solved)
# TODO: implement

# Step 11 - partition_indices (not yet solved)
# TODO: implement

# Step 12 - subset_xy (not yet solved)
# TODO: implement

# Step 13 - ols_fit (not yet solved)
# TODO: implement

# Step 14 - ols_predict (not yet solved)
# TODO: implement

# Step 15 - mean_absolute_error (not yet solved)
# TODO: implement

# Step 16 - root_mean_squared_error (not yet solved)
# TODO: implement

# Step 17 - r_squared (not yet solved)
# TODO: implement

# Step 18 - residual_summary (not yet solved)
# TODO: implement

# Step 19 - prepare_cleaned_features (not yet solved)
# TODO: implement

# Step 20 - assemble_feature_matrix (not yet solved)
# TODO: implement

# Step 21 - make_train_val_test (not yet solved)
# TODO: implement

# Step 22 - standardize_and_add_bias (not yet solved)
# TODO: implement

# Step 23 - evaluate_predictions (not yet solved)
# TODO: implement

# Step 24 - house_price_pipeline (not yet solved)
# TODO: implement

