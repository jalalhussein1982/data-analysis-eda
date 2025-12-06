"""Normality testing functions for distribution analysis."""

from typing import Tuple, Dict
import numpy as np
from scipy import stats


def run_normality_test(data: np.ndarray, alpha: float = 0.05) -> Dict:
    """
    Run normality test on data.

    Uses Shapiro-Wilk for n <= 5000, Kolmogorov-Smirnov otherwise.

    Args:
        data: Numerical data array (NaN values will be removed)
        alpha: Significance level (default 0.05)

    Returns:
        Dictionary with test results:
        - test_name: Name of the test used
        - statistic: Test statistic
        - p_value: P-value
        - is_normal: Boolean indicating if data appears normal
        - interpretation: Human-readable interpretation
    """
    # Remove NaN values
    clean_data = data[~np.isnan(data)]

    if len(clean_data) < 3:
        return {
            "test_name": "N/A",
            "statistic": None,
            "p_value": None,
            "is_normal": None,
            "interpretation": "Insufficient data for normality test (n < 3)"
        }

    # Select test based on sample size
    if len(clean_data) <= 5000:
        test_name = "Shapiro-Wilk"
        statistic, p_value = stats.shapiro(clean_data)
        stat_label = "W-statistic"
    else:
        test_name = "Kolmogorov-Smirnov"
        statistic, p_value = stats.kstest(clean_data, 'norm',
                                           args=(np.mean(clean_data), np.std(clean_data)))
        stat_label = "D-statistic"

    is_normal = p_value >= alpha

    if is_normal:
        interpretation = (
            f"Cannot reject normality (p >= {alpha}). "
            "Pearson correlation is appropriate for this variable."
        )
    else:
        interpretation = (
            f"Reject normality hypothesis (p < {alpha}). "
            "Consider using Spearman's rho (rank-based) or transform the variable."
        )

    return {
        "test_name": test_name,
        "stat_label": stat_label,
        "statistic": statistic,
        "p_value": p_value,
        "is_normal": is_normal,
        "interpretation": interpretation
    }


def get_skewness_label(skewness: float) -> Tuple[str, str]:
    """
    Get label and color code for skewness value.

    Args:
        skewness: Skewness value

    Returns:
        Tuple of (label, color_emoji)
    """
    abs_skew = abs(skewness)

    if abs_skew <= 0.5:
        return "approximately symmetric", "green"
    elif abs_skew <= 1.0:
        direction = "right" if skewness > 0 else "left"
        return f"moderately {direction}-skewed", "yellow"
    else:
        direction = "right" if skewness > 0 else "left"
        return f"highly {direction}-skewed", "red"


def get_kurtosis_label(kurtosis: float) -> Tuple[str, str]:
    """
    Get label and color code for kurtosis value.

    Args:
        kurtosis: Kurtosis value (excess kurtosis, where normal = 0)

    Returns:
        Tuple of (label, color)
    """
    if abs(kurtosis) <= 1:
        return "normal tails (mesokurtic)", "green"
    elif kurtosis > 1:
        return "heavy tails (leptokurtic)", "yellow"
    else:
        return "light tails (platykurtic)", "yellow"


def compute_distribution_stats(data: np.ndarray) -> Dict:
    """
    Compute comprehensive distribution statistics.

    Args:
        data: Numerical data array

    Returns:
        Dictionary with all distribution statistics
    """
    clean_data = data[~np.isnan(data)]

    if len(clean_data) == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "mode": None,
            "std": None,
            "iqr": None,
            "range": None,
            "min": None,
            "max": None,
            "skewness": None,
            "kurtosis": None,
            "q1": None,
            "q3": None
        }

    # Try to find mode
    try:
        mode_result = stats.mode(clean_data, keepdims=True)
        mode_val = mode_result.mode[0]
    except:
        mode_val = None

    q1 = np.percentile(clean_data, 25)
    q3 = np.percentile(clean_data, 75)

    return {
        "count": len(clean_data),
        "mean": np.mean(clean_data),
        "median": np.median(clean_data),
        "mode": mode_val,
        "std": np.std(clean_data),
        "iqr": q3 - q1,
        "range": np.max(clean_data) - np.min(clean_data),
        "min": np.min(clean_data),
        "max": np.max(clean_data),
        "skewness": stats.skew(clean_data),
        "kurtosis": stats.kurtosis(clean_data),
        "q1": q1,
        "q3": q3
    }
