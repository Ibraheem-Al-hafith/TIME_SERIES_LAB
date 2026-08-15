"""Decoupled statistical diagnostic and fitting engine for time series analysis.

Provides strongly typed containers and pure functions for stationarity checks
(ADF and KPSS tests), decomposition, and model fitting.
"""

import logging
import warnings
from dataclasses import dataclass
from typing import Dict, Literal, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TestResult:
    """Immutable data container for individual statistical hypothesis test outputs.

    Attributes:
        statistic: Test statistic value.
        p_value: Calculated p-value for the test.
        lags_used: Number of lags utilized in calculation.
        critical_values: Dictionary mapping significance levels to critical values.
        is_stationary: Boolean evaluation based on significance level (alpha).
        message: Diagnostic string describing execution or statistical context.
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
        observations: Count of valid non-null observations in the series.
        is_stationary: Combined judgment indicating if series is strictly stationary.
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


@dataclass
class FitResult:
    """Container for in-sample model fit outputs.

    Attributes:
        fitted_values: Time series of in-sample model predictions.
        metrics_df: Dataframe summarizing goodness-of-fit metrics.
        success: Boolean flag for model fitting execution status.
        error_message: Detailed message in case of model fitting failure.
    """

    fitted_values: pd.Series
    metrics_df: pd.DataFrame
    success: bool
    error_message: str = ""


def _run_adf_test(clean_series: pd.Series, alpha: float) -> TestResult:
    """Internal helper to execute Augmented Dickey-Fuller test."""
    try:
        res = adfuller(clean_series, autolag="AIC")
        crit_map = {str(k): float(v) for k, v in res[4].items()}
        p_val = float(res[1])
        # ADF H0: Series has a unit root (non-stationary).
        # Rejection (p_val < alpha) implies stationarity.
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
    clean_series: pd.Series, alpha: float, regression: str = "c"
) -> TestResult:
    """Internal helper to execute KPSS test suppressing interpolation warnings."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", InterpolationWarning)
            res = kpss(clean_series, regression=regression, nlags="auto")

        crit_map = {str(k): float(v) for k, v in res[3].items()}
        p_val = float(res[1])
        # KPSS H0: Series is trend/level stationary.
        # Fail to reject (p_val >= alpha) implies stationarity.
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


def calculate_stationarity(
    series: pd.Series, alpha: float = 0.05, kpss_regression: str = "c"
) -> CombinedStationarityResult:
    """Calculates ADF and KPSS stationarity diagnostics over a time series vector.

    Executes both Augmented Dickey-Fuller (ADF) and Kwiatkowski-Phillips-Schmidt-Shin
    (KPSS) tests to provide a robust joint determination of stationarity.

    Args:
        series: Target time series data vector.
        alpha: Significance threshold for statistical decision boundary.
        kpss_regression: 'c' for stationarity around a constant (default) or 'ct' for
            trend stationarity.

    Returns:
        CombinedStationarityResult detailing ADF results, KPSS results, joint decision,
        and diagnostic commentary.
    """
    clean_series = series.dropna()
    obs_count = len(clean_series)

    # Pre-validation fallback result
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

    if np.all(clean_series == clean_series.iloc[0]):
        return _fallback_result("Rejected: Series contains zero variance (constant value).")

    adf_res = _run_adf_test(clean_series, alpha=alpha)
    kpss_res = _run_kpss_test(clean_series, alpha=alpha, regression=kpss_regression)

    # Evaluate joint hypothesis logic
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


# Backward compatibility alias
calculate_adf_stationarity = calculate_stationarity

def generate_stationarity_report(result: CombinedStationarityResult) -> str:
    """Generates a structured Markdown report for stationarity diagnostic results.

    Args:
        result: The combined result object containing ADF and KPSS test data.

    Returns:
        A formatted Markdown string for UI display.
    """
    # Helper to map boolean to emoji/text for quick visual scanning
    def _fmt_bool(val: bool) -> str:
        return "✅ TRUE" if val else "❌ FALSE"
    report = f"""### Time Series Stationarity Audit
* **Overall Conclusion**: **{result.conclusion_type}**
* **Observations Analyzed**: `{result.observations}`
* **Message**: {result.message}
"""
    report += f"""\n---
#### Augmented Dickey-Fuller (ADF) Test
*Null Hypothesis ($H_0$): Series has a unit root (Non-Stationary).*
* **Is Stationary?**: {_fmt_bool(result.adf.is_stationary)}
* **Statistic Value**: `{result.adf.statistic:.4f}`
* **p-Value Probability**: `{result.adf.p_value:.6f}`
"""
    

    report += f"""\n#### Kwiatkowski-Phillips-Schmidt-Shin (KPSS) Test
*Null Hypothesis ($H_0$): Series is trend/level stationary.*
* **Is Stationary?**: {_fmt_bool(result.kpss.is_stationary)}
* **Statistic Value**: `{result.kpss.statistic:.4f}`
* **p-Value Probability**: `{result.kpss.p_value:.6f}`
"""
    
    for pct, val in result.kpss.critical_values.items():
        report += f"\n* **Critical Boundary ({pct})**: `{val:.4f}`"
    
    return report

# Usage in your UI loop:
# stationarity_data = calculate_stationarity(differenced_series)
# ui_markdown = generate_stationarity_report(stationarity_data)

def fit_decomposition(series: pd.Series, model: str, period: int) -> FitResult:
    """Computes seasonal decomposition in-sample fit values and training metrics.

    Args:
        series: Input time series.
        model: Decomposition model type ('additive' or 'multiplicative').
        period: Seasonal period length.

    Returns:
        FitResult container with fitted values and error metrics.
    """
    clean_series = series.dropna()
    if len(clean_series) < (2 * period):
        period = max(2, len(clean_series) // 2 - 1)

    try:
        decomp = seasonal_decompose(
            clean_series, model=model, period=period, extrapolate_trend="freq"
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
        return FitResult(fitted_values=in_sample_preds, metrics_df=metrics_df, success=True)
    except Exception as exc:
        logger.error("Decomposition fit failed: %s", exc)
        return FitResult(
            fitted_values=pd.Series(dtype=float),
            metrics_df=pd.DataFrame([{"Metric": "Error Encountered", "Value": str(exc)}]),
            success=False,
            error_message=str(exc),
        )


def fit_holt_winters(
    series: pd.Series, trend: Optional[str], seasonal: Optional[str], period: int
) -> FitResult:
    """Fits Holt-Winters Exponential Smoothing model and computes diagnostics.

    Args:
        series: Target time series.
        trend: Trend component specification ('add', 'mul', or None/'None').
        seasonal: Seasonal component specification ('add', 'mul', or None/'None').
        period: Seasonal period length.

    Returns:
        FitResult container containing fitted predictions and goodness-of-fit metrics.
    """
    clean_series = series.dropna()
    t_mode = None if trend in (None, "None") else trend
    s_mode = None if seasonal in (None, "None") else seasonal

    try:
        model = ExponentialSmoothing(
            clean_series,
            trend=t_mode,
            seasonal=s_mode,
            seasonal_periods=period if s_mode else None,
            initialization_method="estimated",
        )
        fitted = model.fit()
        in_sample_preds = fitted.fittedvalues

        mae = float(np.mean(np.abs(clean_series - in_sample_preds)))
        rmse = float(np.sqrt(np.mean((clean_series - in_sample_preds) ** 2)))

        metrics_df = pd.DataFrame(
            [
                {"Metric": "Training MAE", "Value": f"{mae:.4f}"},
                {"Metric": "Training RMSE", "Value": f"{rmse:.4f}"},
                {"Metric": "AIC", "Value": f"{fitted.aic:.2f}"},
                {"Metric": "BIC", "Value": f"{fitted.bic:.2f}"},
            ]
        )
        return FitResult(fitted_values=in_sample_preds, metrics_df=metrics_df, success=True)
    except Exception as exc:
        logger.error("Holt-Winters fit failed: %s", exc)
        return FitResult(
            fitted_values=pd.Series(dtype=float),
            metrics_df=pd.DataFrame([{"Metric": "Error Encountered", "Value": str(exc)}]),
            success=False,
            error_message=str(exc),
        )


def fit_sarima(
    series: pd.Series, p: int, d: int, q: int, P: int, D: int, Q: int, period: int
) -> FitResult:
    """Fits SARIMA model and returns in-sample predictions and metrics.

    Args:
        series: Target time series vector.
        p: Non-seasonal AR order.
        d: Non-seasonal differencing degree.
        q: Non-seasonal MA order.
        P: Seasonal AR order.
        D: Seasonal differencing degree.
        Q: Seasonal MA order.
        period: Seasonal period length.

    Returns:
        FitResult container containing fitted predictions and metric outputs.
    """
    clean_series = series.dropna()

    try:
        model = sm.tsa.statespace.SARIMAX(
            clean_series,
            order=(p, d, q),
            seasonal_order=(P, D, Q, period),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fitted = model.fit(disp=False)
        in_sample_preds = fitted.fittedvalues

        mae = float(np.mean(np.abs(clean_series - in_sample_preds)))
        rmse = float(np.sqrt(np.mean((clean_series - in_sample_preds) ** 2)))

        metrics_df = pd.DataFrame(
            [
                {"Metric": "Training MAE", "Value": f"{mae:.4f}"},
                {"Metric": "Training RMSE", "Value": f"{rmse:.4f}"},
                {"Metric": "AIC", "Value": f"{fitted.aic:.2f}"},
                {"Metric": "BIC", "Value": f"{fitted.bic:.2f}"},
            ]
        )
        return FitResult(fitted_values=in_sample_preds, metrics_df=metrics_df, success=True)
    except Exception as exc:
        logger.error("SARIMA fit failed: %s", exc)
        return FitResult(
            fitted_values=pd.Series(dtype=float),
            metrics_df=pd.DataFrame([{"Metric": "Error Encountered", "Value": str(exc)}]),
            success=False,
            error_message=str(exc),
        )