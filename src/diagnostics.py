"""Decoupled statistical diagnostic and fitting engine for time series analysis.

Provides strongly typed containers and pure functions for stationarity checks,
differencing operations, decomposition, and model fitting.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ADFResult:
    """Immutable data container holding the results of an Augmented Dickey-Fuller test."""
    statistic: float
    p_value: float
    lags_used: int
    observations: int
    critical_values: Dict[str, float]
    is_stationary: bool
    message: str


@dataclass
class FitResult:
    """Container for in-sample model fit outputs."""
    fitted_values: pd.Series
    metrics_df: pd.DataFrame
    success: bool
    error_message: str = ""


def calculate_adf_stationarity(series: pd.Series, alpha: float = 0.05) -> ADFResult:
    """Computes stateless Augmented Dickey-Fuller parameters over a data vector."""
    clean_series = series.dropna()
    obs_count = len(clean_series)

    if obs_count < 10:
        return ADFResult(
            statistic=0.0, p_value=1.0, lags_used=0, observations=obs_count,
            critical_values={"1%": 0.0, "5%": 0.0, "10%": 0.0}, is_stationary=False,
            message=f"Rejected: Series length ({obs_count}) contains insufficient entries."
        )

    if np.all(clean_series == clean_series.iloc[0]):
        return ADFResult(
            statistic=0.0, p_value=1.0, lags_used=0, observations=obs_count,
            critical_values={"1%": 0.0, "5%": 0.0, "10%": 0.0}, is_stationary=False,
            message="Rejected: Constant array structure detected (zero variance)."
        )

    try:
        res = adfuller(clean_series, autolag="AIC")
        crit_map = {str(k): float(v) for k, v in res[4].items()}
        p_val = float(res[1])

        return ADFResult(
            statistic=float(res[0]), p_value=p_val, lags_used=int(res[2]),
            observations=int(res[3]), critical_values=crit_map, is_stationary=(p_val < alpha),
            message="Success: Stationarity calculation completed normally."
        )
    except Exception as exc:
        logger.warning("Internal statsmodels adfuller operation failed: %s", exc)
        return ADFResult(
            statistic=0.0, p_value=1.0, lags_used=0, observations=obs_count,
            critical_values={"1%": 0.0, "5%": 0.0, "10%": 0.0}, is_stationary=False,
            message=f"Execution Fault: {str(exc)}"
        )


def fit_decomposition(series: pd.Series, model: str, period: int) -> FitResult:
    """Computes decomposition in-sample fit values and training metrics."""
    clean_series = series.dropna()
    if len(clean_series) < (2 * period):
        period = max(2, len(clean_series) // 2 - 1)

    try:
        decomp = seasonal_decompose(clean_series, model=model, period=period, extrapolate_trend="freq")
        
        if model == "multiplicative":
            in_sample_preds = decomp.trend * decomp.seasonal
        else:
            in_sample_preds = decomp.trend + decomp.seasonal

        mae = np.mean(np.abs(clean_series - in_sample_preds))
        rmse = np.sqrt(np.mean((clean_series - in_sample_preds) ** 2))
        
        metrics_df = pd.DataFrame([
            {"Metric": "Training MAE", "Value": f"{mae:.4f}"},
            {"Metric": "Training RMSE", "Value": f"{rmse:.4f}"}
        ])
        return FitResult(fitted_values=in_sample_preds, metrics_df=metrics_df, success=True)
    except Exception as exc:
        return FitResult(
            fitted_values=pd.Series(dtype=float),
            metrics_df=pd.DataFrame([{"Metric": "Error Encountered", "Value": str(exc)}]),
            success=False,
            error_message=str(exc)
        )


def fit_holt_winters(series: pd.Series, trend: str, seasonal: str, period: int) -> FitResult:
    """Fits Holt-Winters model and returns in-sample predictions and metrics."""
    clean_series = series.dropna()
    t_mode = None if trend == "None" else trend
    s_mode = None if seasonal == "None" else seasonal

    try:
        model = ExponentialSmoothing(
            clean_series,
            trend=t_mode,
            seasonal=s_mode,
            seasonal_periods=period if s_mode else None,
            initialization_method="estimated"
        )
        fitted = model.fit()
        in_sample_preds = fitted.fittedvalues

        mae = np.mean(np.abs(clean_series - in_sample_preds))
        rmse = np.sqrt(np.mean((clean_series - in_sample_preds) ** 2))
        
        metrics_df = pd.DataFrame([
            {"Metric": "Training MAE", "Value": f"{mae:.4f}"},
            {"Metric": "Training RMSE", "Value": f"{rmse:.4f}"},
            {"Metric": "AIC", "Value": f"{fitted.aic:.2f}"},
            {"Metric": "BIC", "Value": f"{fitted.bic:.2f}"}
        ])
        return FitResult(fitted_values=in_sample_preds, metrics_df=metrics_df, success=True)
    except Exception as exc:
        return FitResult(
            fitted_values=pd.Series(dtype=float),
            metrics_df=pd.DataFrame([{"Metric": "Error Encountered", "Value": str(exc)}]),
            success=False,
            error_message=str(exc)
        )


def fit_sarima(
    series: pd.Series, p: int, d: int, q: int, P: int, D: int, Q: int, period: int
) -> FitResult:
    """Fits SARIMA model and returns in-sample predictions and metrics."""
    clean_series = series.dropna()

    try:
        model = sm.tsa.statespace.SARIMAX(
            clean_series,
            order=(p, d, q),
            seasonal_order=(P, D, Q, period),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        fitted = model.fit(disp=False)
        in_sample_preds = fitted.fittedvalues

        mae = np.mean(np.abs(clean_series - in_sample_preds))
        rmse = np.sqrt(np.mean((clean_series - in_sample_preds) ** 2))
        
        metrics_df = pd.DataFrame([
            {"Metric": "Training MAE", "Value": f"{mae:.4f}"},
            {"Metric": "Training RMSE", "Value": f"{rmse:.4f}"},
            {"Metric": "AIC", "Value": f"{fitted.aic:.2f}"},
            {"Metric": "BIC", "Value": f"{fitted.bic:.2f}"}
        ])
        return FitResult(fitted_values=in_sample_preds, metrics_df=metrics_df, success=True)
    except Exception as exc:
        return FitResult(
            fitted_values=pd.Series(dtype=float),
            metrics_df=pd.DataFrame([{"Metric": "Error Encountered", "Value": str(exc)}]),
            success=False,
            error_message=str(exc)
        )