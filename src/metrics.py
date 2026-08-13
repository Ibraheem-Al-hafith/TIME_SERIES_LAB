"""Modular, validation-driven mathematical processing layer for forecast evaluation.

This module provides highly optimized, vectorized numpy metrics and a central
ScoringEngine that handles data alignment, missing values, and configuration flags.
"""

import logging
from .config import ScoringConfig
from typing import Dict, Optional, Set, Tuple

import numpy as np
import pandas as pd

# Setup module-level logger
logger = logging.getLogger(__name__)


# =====================================================================
# CUSTOM EXCEPTIONS DEFINITION
# =====================================================================

class MetricsError(Exception):
    """Base exception class for all evaluation metric processing errors."""
    pass


class AlignmentError(MetricsError):
    """Raised when shapes or temporal dimensions fail to map structural constraints."""
    pass


class MathematicalDomainError(MetricsError):
    """Raised when numerical violations occur (e.g., division by zero)."""
    pass


# =====================================================================
# VECTORIZED PURE MATHEMATICAL FUNCTIONS
# =====================================================================

def calculate_mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculates Mean Absolute Error using optimized numpy vector operations.

    Formula:
        $$MAE = \\frac{1}{n} \\sum_{i=1}^{n} |y_i - \\hat{y}_i|$$

    Args:
        actual: Clean vector containing target ground-truth elements.
        predicted: Clean vector containing forecasted elements.

    Returns:
        The calculated metric score as a plain float value.
    """
    return float(np.mean(np.abs(actual - predicted)))


def calculate_mse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculates Mean Squared Error using optimized numpy vector operations.

    Formula:
        $$MSE = \\frac{1}{n} \\sum_{i=1}^{n} (y_i - \\hat{y}_i)^2$$

    Args:
        actual: Clean vector containing target ground-truth elements.
        predicted: Clean vector containing forecasted elements.

    Returns:
        The calculated metric score as a plain float value.
    """
    return float(np.mean((actual - predicted) ** 2))


def calculate_rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculates Root Mean Squared Error using optimized numpy vector operations.

    Formula:
        $$RMSE = \\sqrt{\\frac{1}{n} \\sum_{i=1}^{n} (y_i - \\hat{y}_i)^2}$$

    Args:
        actual: Clean vector containing target ground-truth elements.
        predicted: Clean vector containing forecasted elements.

    Returns:
        The calculated metric score as a plain float value.
    """
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def calculate_mape(actual: np.ndarray, predicted: np.ndarray, epsilon: Optional[float] = None) -> float:
    """Calculates Mean Absolute Percentage Error with dynamic zero protection logic.

    Formula:
        $$MAPE = \\frac{100\\%}{n} \\sum_{i=1}^{n} \\left| \\frac{y_i - \\hat{y}_i}{y_i} \\right|$$

    Args:
        actual: Clean vector containing target ground-truth elements.
        predicted: Clean vector containing forecasted elements.
        epsilon: Minimum numeric floor scale allowed for targets. If None,
            the function triggers a strict MathematicalDomainError on true zeros.

    Returns:
        The calculated metric score as a percentage value (0-100%).

    Raises:
        MathematicalDomainError: If an actual ground-truth element is zero
            and no epsilon fallback strategy is defined.
    """
    zero_mask = (actual == 0.0)
    
    if np.any(zero_mask):
        if epsilon is None:
            raise MathematicalDomainError(
                "MAPE calculation failed: Actual target matrix contains true zero values. "
                "Configure an 'epsilon' threshold value inside your configuration block "
                "to execute soft numerical correction boundaries."
            )
        logger.warning(
            "Zero values identified in MAPE ground truth. Applying soft epsilon threshold "
            "adjustment floor value: %e", epsilon
        )
        # Create a local copy to safeguard input array from mutations
        actual_adjusted = actual.copy()
        actual_adjusted[zero_mask] = epsilon
        return float(np.mean(np.abs((actual_adjusted - predicted) / actual_adjusted)) * 100.0)

    return float(np.mean(np.abs((actual - predicted) / actual)) * 100.0)


# =====================================================================
# CORE ORCHESTRATION RUNNER ENGINE
# =====================================================================

class ScoringEngine:
    """Validates alignment properties, strips missing indexes, and computes metric portfolios."""

    def __init__(self, config: ScoringConfig) -> None:
        """Initializes the engine with an immutable validation parameter configuration block.

        Args:
            config: A populated ScoringConfig container instance.
        """
        self.config: ScoringConfig = config
        logger.debug("ScoringEngine configured with active payload metrics matrix: %s", config)

    def _sanitize_and_align_inputs(self, actuals: pd.Series, predictions: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """Runs structural validation and drops missing NaN slices securely.

        Args:
            actuals: Evaluated real target timeline series data.
            predictions: Projected out-of-sample forecast series data.

        Returns:
            A tuple of two sanitized, aligned numpy arrays [y_true, y_pred].

        Raises:
            AlignmentError: If lengths vary or if calendar indexing maps do not align.
        """
        if len(actuals) != len(predictions):
            raise AlignmentError(
                f"Evaluation size mismatch error. Actual ground truth length ({len(actuals)}) "
                f"does not match prediction horizon length ({len(predictions)})."
            )

        if not actuals.index.equals(predictions.index):
            # Locate set differences for trace metrics mapping
            mismatched_dates: Set = set(actuals.index).symmetric_difference(set(predictions.index))
            raise AlignmentError(
                f"Temporal datetime alignment error. Index maps do not match exactly. "
                f"Sample mismatched context size: {min(len(mismatched_dates), 5)} points. "
                f"Check frequency or boundary offsets."
            )

        # Detect and flag missing NaN items dynamically across both inputs
        combined_nan_mask = actuals.isna() | predictions.isna()
        
        if combined_nan_mask.any():
            nan_count = int(combined_nan_mask.sum())
            logger.warning(
                "Identified %d missing NaN data element(s) within the evaluation window. "
                "Performing automatic structural alignment drop.", nan_count
            )
            y_true = actuals.loc[~combined_nan_mask].to_numpy(dtype=np.float64)
            y_pred = predictions.loc[~combined_nan_mask].to_numpy(dtype=np.float64)
        else:
            y_true = actuals.to_numpy(dtype=np.float64)
            y_pred = predictions.to_numpy(dtype=np.float64)

        if len(y_true) == 0:
            raise AlignmentError(
                "Input processing failed: Complete array series dropped due to internal NaN masks."
            )

        return y_true, y_pred

    def evaluate(self, actuals: pd.Series, predictions: pd.Series) -> Dict[str, float]:
        """Calculates configured statistical scores across input series.

        Args:
            actuals: Evaluated real target timeline series data.
            predictions: Projected out-of-sample forecast series data.

        Returns:
            A dictionary containing structural metric keys mapped to computed score floats.
        """
        # 1. Enforce alignment validations and extract sanitized data arrays
        y_true, y_pred = self._sanitize_and_align_inputs(actuals, predictions)
        
        performance_report: Dict[str, float] = {}

        # 2. Process metrics dynamically using explicit configuration criteria
        if self.config.mae:
            performance_report["MAE"] = calculate_mae(y_true, y_pred)
            
        if self.config.mse:
            performance_report["MSE"] = calculate_mse(y_true, y_pred)
            
        if self.config.rmse:
            performance_report["RMSE"] = calculate_rmse(y_true, y_pred)
            
        if self.config.mape:
            performance_report["MAPE"] = calculate_mape(y_true, y_pred, epsilon=self.config.epsilon)

        logger.info("Evaluation performance run completed successfully. Active metrics count: %d", len(performance_report))
        return performance_report