"""Decoupled statistical diagnostic and time-series decomposition audit engine.

Provides strongly typed immutable containers and functions for stationarity checks
(ADF and KPSS tests), statistical decomposition diagnostics, and Markdown report generation.
This module is strictly designed for exploratory data analysis and time-series profiling,
delegating predictive model training exclusively to the forecasting models suite.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Dict, Literal, Optional

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss

from .data import DataClass

logger = logging.getLogger(__name__)


# =====================================================================
# DATA CONTAINERS
# =====================================================================

@dataclass(frozen=True)
class TestResult:
    """Immutable data container for individual statistical hypothesis test outputs.

    Attributes:
        statistic: Calculated test statistic value.
        p_value: Calculated p-value for the hypothesis test.
        lags_used: Number of lag orders utilized in the calculation.
        critical_values: Dictionary mapping significance thresholds to critical values.
        is_stationary: Boolean evaluation based on significance level (alpha).
        message: Diagnostic string describing execution outcome or statistical context.
    """

    statistic: float
    p_value: float
    lags_used: int
    critical_values: Dict[str, float]
    is_stationary: bool
    message: str


@dataclass(frozen=True)
class CombinedStationarityResult:
    """Immutable container holding unified stationarity analysis (ADF + KPSS).

    Attributes:
        adf: Detailed results from Augmented Dickey-Fuller test.
        kpss: Detailed results from Kwiatkowski-Phillips-Schmidt-Shin test.
        observations: Count of valid non-null observations in evaluated series.
        is_stationary: Combined judgment indicating whether series is strictly stationary.
        conclusion_type: Categorical conclusion based on joint hypothesis tests.
        message: Consolidated diagnostic summary.
    """

    adf: TestResult
    kpss: TestResult
    observations: int
    is_stationary: bool
    conclusion_type: Literal[
        "Strictly Stationary",
        "Difference Stationary",
        "Trend Stationary",
        "Non-Stationary",
        "Insufficient Data / Error",
    ]
    message: str


@dataclass(frozen=True)
class DecompositionFitResult:
    """Immutable container for in-sample structural decomposition diagnostics.

    Attributes:
        fitted_values: Time series of in-sample reconstructed signal predictions.
        metrics_df: DataFrame summarizing goodness-of-fit training metrics (MAE, RMSE).
        success: Boolean flag for decomposition execution status.
        error_message: Detailed exception message in case of failure.
    """

    fitted_values: pd.Series
    metrics_df: pd.DataFrame
    success: bool
    error_message: str = ""


# =====================================================================
# HYPOTHESIS TESTING HELPERS
# =====================================================================

def _run_adf_test(clean_series: pd.Series, alpha: float) -> TestResult:
    """Execute Augmented Dickey-Fuller (ADF) unit root test.

    Null Hypothesis (H0): Series possesses a unit root (Non-Stationary).
    Alternative (H1): Series is stationary.

    Args:
        clean_series: Target vector stripped of missing values.
        alpha: Decision threshold significance level.

    Returns:
        TestResult detailing numerical outputs and stationary determination.
    """
    try:
        res = adfuller(clean_series, autolag="AIC")
        assert type(res[4]) is dict
        crit_map = {str(k): float(v) for k, v in res[4].items()}
        p_val = float(res[1])
        # Rejection of H0 (p_val < alpha) implies stationarity.
        is_stationary = p_val < alpha

        return TestResult(
            statistic=float(res[0]),
            p_value=p_val,
            lags_used=int(res[2]),
            critical_values=crit_map,
            is_stationary=is_stationary,
            message="ADF test executed successfully.",
        )
    except Exception as exc:
        logger.warning("ADF test execution failed: %s", exc)
        return TestResult(
            statistic=0.0,
            p_value=1.0,
            lags_used=0,
            critical_values={"1%": 0.0, "5%": 0.0, "10%": 0.0},
            is_stationary=False,
            message=f"ADF Execution Fault: {exc}",
        )


def _run_kpss_test(
    clean_series: pd.Series, alpha: float, regression: Literal['c', 'ct'] = 'c'
) -> TestResult:
    """Execute Kwiatkowski-Phillips-Schmidt-Shin (KPSS) stationarity test.

    Null Hypothesis (H0): Series is trend/level stationary.
    Alternative (H1): Series has a unit root (Non-Stationary).

    Args:
        clean_series: Target vector stripped of missing values.
        alpha: Decision threshold significance level.
        regression: 'c' for level stationarity, 'ct' for trend stationarity.

    Returns:
        TestResult detailing numerical outputs and stationary determination.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", InterpolationWarning)
            res = kpss(clean_series, regression=regression, nlags="auto")

        crit_map = {str(k): float(v) for k, v in res[3].items()}
        p_val = float(res[1])
        # Failure to reject H0 (p_val >= alpha) implies stationarity.
        is_stationary = p_val >= alpha

        return TestResult(
            statistic=float(res[0]),
            p_value=p_val,
            lags_used=int(res[2]),
            critical_values=crit_map,
            is_stationary=is_stationary,
            message="KPSS test executed successfully.",
        )
    except Exception as exc:
        logger.warning("KPSS test execution failed: %s", exc)
        return TestResult(
            statistic=0.0,
            p_value=0.0,
            lags_used=0,
            critical_values={"1%": 0.0, "5%": 0.0, "10%": 0.0},
            is_stationary=False,
            message=f"KPSS Execution Fault: {exc}",
        )


# =====================================================================
# PUBLIC DIAGNOSTIC API
# =====================================================================

def calculate_stationarity(
    data_input: pd.Series | DataClass,
    target_col: Optional[str] = None,
    alpha: float = 0.05,
    kpss_regression: Literal['c', 'ct'] = "c",
) -> CombinedStationarityResult:
    """Perform joint ADF and KPSS stationarity hypothesis analysis over a time series.

    Combines Dickey-Fuller and KPSS tests to resolve stationarity type:
    Strictly Stationary, Difference Stationary, Trend Stationary, or Non-Stationary.

    Args:
        data_input: Source vector as `pd.Series` or injected `DataClass` container.
        target_col: Column name identifier if data_input is a `DataClass`.
        alpha: Significance boundary threshold.
        kpss_regression: 'c' for constant level or 'ct' for trend stationarity.

    Returns:
        CombinedStationarityResult detailing sub-test statistics and joint conclusion.
    """
    # Resolve vector input from DataClass or Series abstraction
    if isinstance(data_input, DataClass):
        col = target_col or getattr(getattr(data_input, "config", None), "target", None)
        if not col or col not in data_input.train.columns:
            raise ValueError(
                f"Target column '{col}' is invalid or missing in training dataset."
            )
        series = data_input.train[col]
    elif isinstance(data_input, pd.Series):
        series = data_input
    else:
        raise TypeError(
            f"Unsupported input payload type '{type(data_input)}'. Expected DataClass or pd.Series."
        )

    clean_series = series.dropna()
    obs_count = len(clean_series)

    def _fallback_result(msg: str) -> CombinedStationarityResult:
        empty_test = TestResult(
            statistic=0.0,
            p_value=1.0,
            lags_used=0,
            critical_values={"1%": 0.0, "5%": 0.0, "10%": 0.0},
            is_stationary=False,
            message=msg,
        )
        return CombinedStationarityResult(
            adf=empty_test,
            kpss=empty_test,
            observations=obs_count,
            is_stationary=False,
            conclusion_type="Insufficient Data / Error",
            message=msg,
        )

    if obs_count < 10:
        return _fallback_result(
            f"Rejected: Insufficient observations ({obs_count} < 10)."
        )

    if np.all(clean_series.to_numpy() == clean_series.iloc[0]):
        return _fallback_result(
            "Rejected: Series contains zero variance (constant value)."
        )

    adf_res = _run_adf_test(clean_series, alpha=alpha)
    kpss_res = _run_kpss_test(clean_series, alpha=alpha, regression=kpss_regression)

    # Evaluate joint hypothesis logic matrix
    if adf_res.is_stationary and kpss_res.is_stationary:
        is_stationary = True
        conclusion = "Strictly Stationary"
        msg = "Both tests confirm stationarity. Series is stationary."
    elif adf_res.is_stationary and not kpss_res.is_stationary:
        is_stationary = False
        conclusion = "Difference Stationary"
        msg = "ADF indicates stationary, but KPSS indicates unit root/trend. Consider differencing."
    elif not adf_res.is_stationary and kpss_res.is_stationary:
        is_stationary = True
        conclusion = "Trend Stationary"
        msg = "ADF indicates unit root, but KPSS indicates trend stationarity. Consider detrending."
    else:
        is_stationary = False
        conclusion = "Non-Stationary"
        msg = "Both tests confirm non-stationarity."

    return CombinedStationarityResult(
        adf=adf_res,
        kpss=kpss_res,
        observations=obs_count,
        is_stationary=is_stationary,
        conclusion_type=conclusion,
        message=msg,
    )


def generate_stationarity_report(result: CombinedStationarityResult) -> str:
    """Format combined stationarity diagnostic outputs into a Markdown report.

    Args:
        result: Combined stationarity test results.

    Returns:
        A formatted Markdown string suitable for logs, dashboards, or UI rendering.
    """
    def _fmt_bool(val: bool) -> str:
        return "✅ TRUE" if val else "❌ FALSE"

    report = f"""### Time Series Stationarity Audit
* **Overall Conclusion**: **{result.conclusion_type}**
* **Observations Analyzed**: `{result.observations}`
* **Message**: {result.message}

---
#### Augmented Dickey-Fuller (ADF) Test
*Null Hypothesis ($H_0$): Series has a unit root (Non-Stationary).*
* **Is Stationary?**: {_fmt_bool(result.adf.is_stationary)}
* **Statistic Value**: `{result.adf.statistic:.4f}`
* **p-Value Probability**: `{result.adf.p_value:.6f}`

#### Kwiatkowski-Phillips-Schmidt-Shin (KPSS) Test
*Null Hypothesis ($H_0$): Series is trend/level stationary.*
* **Is Stationary?**: {_fmt_bool(result.kpss.is_stationary)}
* **Statistic Value**: `{result.kpss.statistic:.4f}`
* **p-Value Probability**: `{result.kpss.p_value:.6f}`
"""

    for pct, val in result.kpss.critical_values.items():
        report += f"\n* **Critical Boundary ({pct})**: `{val:.4f}`"

    return report


def fit_decomposition(
    data_input: pd.Series | DataClass,
    target_col: Optional[str] = None,
    model: str = "additive",
    period: Optional[int] = None,
) -> DecompositionFitResult:
    """Compute seasonal decomposition in-sample signal reconstruction for diagnostics.

    Args:
        data_input: Input series vector or `DataClass` container.
        target_col: Column label identifier when using `DataClass`.
        model: Decomposition mode ('additive' or 'multiplicative').
        period: Seasonal period length. If None, inferred or defaulted.

    Returns:
        DecompositionFitResult containing reconstructed signal and error metrics.
    """
    if isinstance(data_input, DataClass):
        col = target_col or getattr(getattr(data_input, "config", None), "target", None)
        if not col or col not in data_input.train.columns:
            raise ValueError(
                f"Target column '{col}' is invalid or missing in training dataset."
            )
        series = data_input.train[col]
    elif isinstance(data_input, pd.Series):
        series = data_input
    else:
        raise TypeError(
            f"Unsupported input payload type '{type(data_input)}'. Expected DataClass or pd.Series."
        )

    clean_series = series.dropna()
    resolved_period = period or 12

    if len(clean_series) < (2 * resolved_period):
        resolved_period = max(2, len(clean_series) // 2)

    try:
        decomp = seasonal_decompose(
            clean_series,
            model=model,
            period=resolved_period,
            extrapolate_trend="freq",
        )

        if model == "multiplicative":
            in_sample_preds = decomp.trend * decomp.seasonal
        else:
            in_sample_preds = decomp.trend + decomp.seasonal

        mae = float(np.mean(np.abs(clean_series - in_sample_preds)))
        rmse = float(np.sqrt(np.mean((clean_series - in_sample_preds) ** 2)))

        metrics_df = pd.DataFrame(
            [
                {"Metric": "Training MAE", "Value": f"{mae:.4f}"},
                {"Metric": "Training RMSE", "Value": f"{rmse:.4f}"},
            ]
        )
        return DecompositionFitResult(
            fitted_values=in_sample_preds, metrics_df=metrics_df, success=True
        )
    except Exception as exc:
        logger.error("Decomposition fit failed: %s", exc)
        return DecompositionFitResult(
            fitted_values=pd.Series(dtype=float),
            metrics_df=pd.DataFrame([{"Metric": "Error Encountered", "Value": str(exc)}]),
            success=False,
            error_message=str(exc),
        )