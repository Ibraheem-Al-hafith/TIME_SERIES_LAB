"""Decoupled statistical diagnostic engine for time series analysis.

Provides strongly typed containers and pure functions for stationarity checks,
differencing operations, and correlation structure identification.
"""

import logging
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Thread-safe headless backend for server execution
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm

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


def calculate_adf_stationarity(series: pd.Series, alpha: float = 0.05) -> ADFResult:
    """Computes stateless Augmented Dickey-Fuller parameters over a data vector."""
    clean_series = series.dropna()
    obs_count = len(clean_series)

    if obs_count < 10:
        return ADFResult(
            statistic=0.0, p_value=1.0, lags_used=0, observations=obs_count,
            critical_values={"1%": 0.0, "5%": 0.0, "10%": 0.0}, is_stationary=False,
            message=f"Rejected: Series sequence length ({obs_count}) contains insufficient entries."
        )

    if np.all(clean_series == clean_series.iloc[0]):
        return ADFResult(
            statistic=0.0, p_value=1.0, lags_used=0, observations=obs_count,
            critical_values={"1%": 0.0, "5%": 0.0, "10%": 0.0}, is_stationary=False,
            message="Rejected: Constant array structure detected. Zero variance."
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


def generate_sandbox_directories(base_path: str = "plots/sandbox") -> None:
    """Ensures temporary cache workspaces exist for sandbox artifact storage."""
    os.makedirs(base_path, exist_ok=True)


def generate_decomposition_plot(series: pd.Series, model: str, period: int) -> Tuple[str, pd.DataFrame]:
    """Generates classical decomposition tracking fit paths and training metrics portfolio."""
    generate_sandbox_directories()
    out_path = "plots/sandbox/decomposition_explorer.png"
    
    clean_series = series.dropna()
    metrics_df = pd.DataFrame(columns=["Metric", "Value"])
    if len(clean_series) < (2 * period):
        period = max(2, len(clean_series) // 2 - 1)
        
    try:
        decomp = seasonal_decompose(clean_series, model=model, period=period, extrapolate_trend="freq")
        
        # In-sample reconstruction formula depending on composition rule
        if model == "multiplicative":
            in_sample_preds = decomp.trend * decomp.seasonal
        else:
            in_sample_preds = decomp.trend + decomp.seasonal
            
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(clean_series.index, clean_series.values, label="Actual (In-Sample)", color="black", linewidth=1.5)
        ax.plot(in_sample_preds.index, in_sample_preds.values, label="Decomposition Fit Path", color="blue", linestyle="--")
        ax.set_title(f"Classical Decomposition Fit Track (Mode={model.capitalize()}, Period={period})")
        ax.legend(loc="upper left")
        ax.grid(True, linestyle=":", alpha=0.6)
        
        plt.tight_layout()
        fig.savefig(out_path, bbox_inches="tight", dpi=100)
        plt.close(fig)
        
        mae = np.mean(np.abs(clean_series - in_sample_preds))
        rmse = np.sqrt(np.mean((clean_series - in_sample_preds) ** 2))
        metrics_df = pd.DataFrame([
            {"Metric": "Training MAE", "Value": f"{mae:.4f}"},
            {"Metric": "Training RMSE", "Value": f"{rmse:.4f}"}
        ])
    except Exception as exc:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, f"Decomposition Error: {str(exc)}", ha="center", va="center", color="red")
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        metrics_df = pd.DataFrame([{"Metric": "Error Encountered", "Value": str(exc)}])
        
    return out_path, metrics_df


def generate_holt_winters_plot(series: pd.Series, trend: Optional[str], seasonal: Optional[str], period: int) -> Tuple[str, pd.DataFrame]:
    """Fits an in-sample Holt-Winters framework state model to trace tracking paths."""
    generate_sandbox_directories()
    out_path = "plots/sandbox/holt_winters_explorer.png"
    
    clean_series = series.dropna()
    metrics_df = pd.DataFrame(columns=["Metric", "Value"])
    
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
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(clean_series.index, clean_series.values, label="Actual (In-Sample)", color="black", linewidth=1.5)
        ax.plot(in_sample_preds.index, in_sample_preds.values, label="Fitted Values", color="orange", linestyle="--")
        ax.set_title(f"Holt-Winters In-Sample Tracking Path (Trend={trend}, Seasonal={seasonal})")
        ax.legend(loc="upper left")
        ax.grid(True, linestyle=":", alpha=0.6)
        
        fig.savefig(out_path, bbox_inches="tight", dpi=100)
        plt.close(fig)
        
        mae = np.mean(np.abs(clean_series - in_sample_preds))
        rmse = np.sqrt(np.mean((clean_series - in_sample_preds) ** 2))
        metrics_df = pd.DataFrame([
            {"Metric": "Training MAE", "Value": f"{mae:.4f}"},
            {"Metric": "Training RMSE", "Value": f"{rmse:.4f}"},
            {"Metric": "AIC", "Value": f"{fitted.aic:.2f}"},
            {"Metric": "BIC", "Value": f"{fitted.bic:.2f}"}
        ])
    except Exception as exc:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, f"Holt-Winters Fitting Fail: {str(exc)}", ha="center", va="center", color="red")
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        metrics_df = pd.DataFrame([{"Metric": "Error Encountered", "Value": str(exc)}])
        
    return out_path, metrics_df


def generate_sarima_sandbox_fit(
    series: pd.Series, p: int, d: int, q: int, P: int, D: int, Q: int, period: int
) -> Tuple[str, pd.DataFrame]:
    """Fits an in-sample SARIMAX state-space matrix layer to trace parameter fit alignment paths."""
    generate_sandbox_directories()
    out_path = "plots/sandbox/sarima_fit_explorer.png"
    clean_series = series.dropna()
    metrics_df = pd.DataFrame(columns=["Metric", "Value"])
    
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
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(clean_series.index, clean_series.values, label="Actual (In-Sample)", color="black", linewidth=1.5)
        ax.plot(in_sample_preds.index, in_sample_preds.values, label="SARIMA In-Sample Fit", color="green", linestyle="--")
        ax.set_title(f"SARIMA Sandbox In-Sample Tracking Path (order=({p},{d},{q}) seasonal_order=({P},{D},{Q},{period}))")
        ax.legend(loc="upper left")
        ax.grid(True, linestyle=":", alpha=0.6)
        
        fig.savefig(out_path, bbox_inches="tight", dpi=100)
        plt.close(fig)
        
        mae = np.mean(np.abs(clean_series - in_sample_preds))
        rmse = np.sqrt(np.mean((clean_series - in_sample_preds) ** 2))
        metrics_df = pd.DataFrame([
            {"Metric": "Training MAE", "Value": f"{mae:.4f}"},
            {"Metric": "Training RMSE", "Value": f"{rmse:.4f}"},
            {"Metric": "AIC", "Value": f"{fitted.aic:.2f}"},
            {"Metric": "BIC", "Value": f"{fitted.bic:.2f}"}
        ])
    except Exception as exc:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, f"SARIMA Fit Failure: {str(exc)}", ha="center", va="center", color="red")
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        metrics_df = pd.DataFrame([{"Metric": "Error Encountered", "Value": str(exc)}])
        
    return out_path, metrics_df


def generate_stateless_correlation_plots(series: pd.Series) -> str:
    """Generates combined twin axis visualizations showing ACF and PACF trends."""
    generate_sandbox_directories()
    out_path = "plots/sandbox/sarima_correlation_diagnostics.png"
    clean_series = series.dropna()
    
    if len(clean_series) < 5:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "Insufficient observations remaining to chart layouts.", ha="center", va="center")
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        return out_path

    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        max_lags = min(24, len(clean_series) // 2 - 1)
        max_lags = max(1, max_lags)
        
        plot_acf(clean_series, ax=axes[0], lags=max_lags, title="Autocorrelation (ACF)")
        plot_pacf(clean_series, ax=axes[1], lags=max_lags, title="Partial Autocorrelation (PACF)", method="ywm")
        
        for ax in axes:
            ax.grid(True, linestyle=":", alpha=0.5)
            
        plt.tight_layout()
        fig.savefig(out_path, bbox_inches="tight", dpi=100)
        plt.close(fig)
    except Exception as exc:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, f"Plotting Error: {str(exc)}", ha="center", va="center", color="red")
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        
    return out_path