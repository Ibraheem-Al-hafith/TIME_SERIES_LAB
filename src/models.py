"""Stateless time series forecasting models suite for production pipelines.

This module provides wrappers around classical decomposition, Holt-Winters 
exponential smoothing, and SARIMA models. It handles hyperparameter configuration
dynamically and safely evaluates multi-frequency temporal data indexes.
"""

import abc
import logging
from .config import (
    DecomposeConfig,
    ExponentialConfig, 
    SARIMAConfig
)
from typing import Any, Dict, Optional, Tuple, TypeVar, Union

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

# Setup module logger
logger = logging.getLogger(__name__)

# Type variable for self-referential class type-hinting
TBaseModel = TypeVar("TBaseModel", bound="BaseModel")


# =====================================================================
# CUSTOM EXCEPTIONS DEFINITION
# =====================================================================

class ForecastingModelError(Exception):
    """Base exception class for all errors inside the forecasting models module."""
    pass


class ModelNotFittedError(ForecastingModelError):
    """Raised when predict() is called before a model is successfully fitted."""
    pass


class DomainValidationError(ForecastingModelError):
    """Raised when mathematical prerequisites or spatial domains are violated."""
    pass


class FrequencyInferenceError(ForecastingModelError):
    """Raised when data index frequency cannot be resolved automatically or manually."""
    pass



ConfigTypes = Union[DecomposeConfig, ExponentialConfig, SARIMAConfig]


# =====================================================================
# BASE CONTRACT STRATEGY
# =====================================================================

class BaseModel(abc.ABC):
    """Abstract interface dictating the structural contract for all forecasting models.

    Enforces stateless configuration execution and dynamic temporal frequency parsing.
    """

    def __init__(self, model_config: ConfigTypes) -> None:
        """Initializes the model wrapper with a read-only configuration payload.

        Args:
            model_config: Config data class containing parameters tailored to the implementation.
        """
        self.config: ConfigTypes = model_config
        self.is_fitted: bool = False
        logger.debug("Initialized %s with config: %s", self.__class__.__name__, model_config)

    def _resolve_frequency_period(self, index: pd.Index, manual_period: Optional[int]) -> int:
        """Inspects the pandas Index structural attributes to locate temporal frequency patterns.

        Maps string tags to numeric integers. Falls back to manual configuration if inference fails.

        Args:
            index: The pandas Index derived from the source training collection.
            manual_period: Explicit period definition overridden via operational config.

        Returns:
            An integer mapping representing the number of observations per cycle.

        Raises:
            FrequencyInferenceError: If frequency cannot be inferred and manual_period is missing.
        """
        if manual_period is not None and manual_period > 0:
            logger.info("Using manually configured seasonal period: %d", manual_period)
            return manual_period

        # Extract frequency properties safely across multiple pandas configurations
        freq_str: Optional[str] = getattr(index, "freqstr", None) or getattr(index, "inferred_freq", None)

        if freq_str is None:
            raise FrequencyInferenceError(
                "Data index lacks explicit temporal frequency metadata. "
                "Ensure pandas index frequency is declared or provide a manual configuration period value."
            )

        # Base Frequency mapping matrix
        frequency_map: Dict[str, int] = {
            "MS": 12, "M": 12,   # Monthly variants
            "QS": 4,  "Q": 4,    # Quarterly variants
            "D": 7,              # Daily variant (weekly seasonal cycle)
            "W": 52              # Weekly variant
        }

        # Substring match to route variants like 'MS', 'M', 'QS', 'Q' accurately
        for key, value in frequency_map.items():
            if key in freq_str:
                logger.info("Inferred seasonal period from index frequency '%s' -> %d", freq_str, value)
                return value

        raise FrequencyInferenceError(
            f"Unsupported frequency pattern identified: '{freq_str}'. "
            f"Provide an explicit integer period within your configuration structure."
        )

    @abc.abstractmethod
    def fit(self, df: pd.DataFrame, target_col: str) -> TBaseModel:
        """Fits the underlying model strategy on the targeted training slice.

        Args:
            df: Ingestion dataframe housing training vectors.
            target_col: Label identifier of the target series column.

        Returns:
            The instance of the fitted model wrapper execution layer.
        """
        pass

    @abc.abstractmethod
    def predict(self, steps: int) -> pd.Series:
        """Projects future timeline values spanning across the requested allocation window.

        Args:
            steps: Integer length of out-of-sample observations to compute.

        Returns:
            A clean pandas Series containing predictions matched with a continuous future DatetimeIndex.

        Raises:
            ModelNotFittedError: If execution is requested before running a successful fit operation.
        """
        pass


# =====================================================================
# CONCRETE IMPLEMENTATION: CLASSICAL DECOMPOSITION FORECASTER
# =====================================================================

class DecompositionModel(BaseModel):
    """Extends statistical decomposition into an active forecasting tool.

    Fits an ordinary least squares linear trend component and loops historical
    seasonal variances forward into the evaluation window.
    """

    def __init__(self, model_config: DecomposeConfig) -> None:
        super().__init__(model_config)
        self.config: DecomposeConfig = model_config
        
        # Internal states populated post-fit
        self.last_training_index: Optional[Any] = None
        self.index_freq: Optional[Any] = None
        self.resolved_period: int = 1
        self.trend_regressor: Optional[LinearRegression] = None
        self.trend_base_length: int = 0
        self.seasonal_cycle_block: Optional[np.ndarray] = None

    def fit(self, df: pd.DataFrame, target_col: str) -> "DecompositionModel":
        if target_col not in df.columns:
            raise KeyError(f"Target column '{target_col}' not found within provided DataFrame columns.")

        series: pd.Series = df[target_col].dropna()
        if series.empty:
            raise ValueError(f"Extracted target series '{target_col}' contains no valid computational elements.")

        # Multiplicative domain safety verification
        if self.config.model == "multiplicative" and (series <= 0).any():
            raise DomainValidationError(
                "Multiplicative decomposition models fail mathematically when encountering values <= 0. "
                "Cleanse training slice or switch configuration model to 'additive'."
            )

        self.last_training_index = series.index[-1]
        
        # Standardize fallback options if explicit frequencies are truncated
        self.index_freq = series.index.freq or pd.tseries.frequencies.to_offset(
            getattr(series.index, "inferred_freq", None)
        )
        
        self.resolved_period = self._resolve_frequency_period(series.index, self.config.period)

        # Decompose the isolated signal array using statsmodels infrastructure
        decomposition = sm.tsa.seasonal_decompose(
            series,
            model=self.config.model,
            period=self.resolved_period
        )

        # Drop missing boundaries safely to generate a clean signal for trend estimation
        valid_indices = ~np.isnan(decomposition.trend)
        clean_trend: pd.Series = decomposition.trend[valid_indices]

        if clean_trend.empty:
            raise DomainValidationError(
                "Insufficient observation window to compute statistical trend components. "
                "Expand training length context."
            )

        # Fit robust ordinary least squares model across calculated trend matrices
        x_indices = np.arange(len(clean_trend)).reshape(-1, 1)
        y_values = clean_trend.values
        
        self.trend_regressor = LinearRegression()
        self.trend_regressor.fit(x_indices, y_values)
        self.trend_base_length = len(series)

        # Isolate exactly one complete seasonal wave from the tail bounds of historical cycles
        self.seasonal_cycle_block = decomposition.seasonal.iloc[-self.resolved_period:].values

        self.is_fitted = True
        return self

    def predict(self, steps: int) -> pd.Series:
        if not self.is_fitted or self.trend_regressor is None or self.seasonal_cycle_block is None:
            raise ModelNotFittedError("Model execution halted: Run fit routine before calling predict paths.")
        
        if steps <= 0:
            raise ValueError("Prediction horizon length steps must be a positive integer greater than 0.")

        # Generate target calendar index markers tracking continuous out-of-sample steps safely
        future_index = pd.date_range(
            start=self.last_training_index + (self.index_freq * 1),
            periods=steps,
            freq=self.index_freq
        )

        # Extrapolate linear trend vector systematically
        trend_steps = np.arange(self.trend_base_length, self.trend_base_length + steps).reshape(-1, 1)
        projected_trend: np.ndarray = self.trend_regressor.predict(trend_steps)

        # Replicate historical seasonal patterns indefinitely across target steps
        repetitions = int(np.ceil(steps / self.resolved_period))
        extended_seasonals = np.tile(self.seasonal_cycle_block, repetitions)[:steps]

        # Combine vector blocks based on configured algebraic model structures
        if self.config.model == "multiplicative":
            forecast_values = projected_trend * extended_seasonals
        else:
            forecast_values = projected_trend + extended_seasonals

        return pd.Series(forecast_values, index=future_index, name="decomposition_forecast")


# =====================================================================
# CONCRETE IMPLEMENTATION: HOLT-WINTERS EXPONENTIAL SMOOTHING
# =====================================================================

class ExponentialSmoothingModel(BaseModel):
    """Wraps statsmodels state space Exponential Smoothing infrastructure."""

    def __init__(self, model_config: ExponentialConfig) -> None:
        super().__init__(model_config)
        self.config: ExponentialConfig = model_config
        self.fitted_results: Optional[Any] = None

    def fit(self, df: pd.DataFrame, target_col: str) -> "ExponentialSmoothingModel":
        if target_col not in df.columns:
            raise KeyError(f"Target column '{target_col}' not found within provided DataFrame columns.")

        series = df[target_col].dropna()
        if series.empty:
            raise ValueError("Extracted target series contains no valid computational elements.")

        # Handle structural validation for multiplicative variations
        is_multiplicative = (self.config.trend == "mul") or (self.config.seasonal == "mul")
        if is_multiplicative and (series <= 0).any():
            raise DomainValidationError(
                "Multiplicative Holt-Winters metrics crash when encountering observations <= 0."
            )

        # Extract context attributes dynamically
        period = self._resolve_frequency_period(series.index, self.config.seasonal_periods)

        # Instantiate implementation engine directly mapping framework specifications
        model_engine = sm.tsa.ExponentialSmoothing(
            series,
            trend=self.config.trend,
            seasonal=self.config.seasonal,
            seasonal_periods=period,
            initialization_method="estimated"
        )

        self.fitted_results = model_engine.fit()
        self.is_fitted = True
        return self

    def predict(self, steps: int) -> pd.Series:
        if not self.is_fitted or self.fitted_results is None:
            raise ModelNotFittedError("Model execution halted: Run fit routine before calling predict paths.")
        
        if steps <= 0:
            raise ValueError("Prediction horizon length steps must be a positive integer greater than 0.")

        forecast = self.fitted_results.forecast(steps=steps)
        return pd.Series(forecast, name="holt_winters_forecast")


# =====================================================================
# CONCRETE IMPLEMENTATION: SEASONAL ARIMA (SARIMAX)
# =====================================================================

class SARIMAModel(BaseModel):
    """Wraps statsmodels SARIMAX modeling framework for comprehensive forecasting."""

    def __init__(self, model_config: SARIMAConfig) -> None:
        super().__init__(model_config)
        self.config: SARIMAConfig = model_config
        self.fitted_results: Optional[Any] = None

    def fit(self, df: pd.DataFrame, target_col: str) -> "SARIMAModel":
        if target_col not in df.columns:
            raise KeyError(f"Target column '{target_col}' not found within provided DataFrame columns.")

        series = df[target_col].dropna()
        if series.empty:
            raise ValueError("Extracted target series contains no valid computational elements.")

        # Intercept temporal structures to resolve seasonal lags securely
        period = self._resolve_frequency_period(series.index, self.config.s)

        # Map vector tuples expected directly by the Statsmodels execution framework
        order_tuple: Tuple[int, int, int] = (self.config.p, self.config.d, self.config.q)
        seasonal_order_tuple: Tuple[int, int, int, int] = (
            self.config.P, self.config.D, self.config.Q, period
        )

        # Construct analytical state-space model pipelines
        model_engine = sm.tsa.statespace.SARIMAX(
            series,
            order=order_tuple,
            seasonal_order=seasonal_order_tuple,
            enforce_stationarity=False,
            enforce_invertibility=False
        )

        # Suppress standard convergence logs to streamline UI execution pipelines
        self.fitted_results = model_engine.fit(disp=False)
        self.is_fitted = True
        return self

    def predict(self, steps: int) -> pd.Series:
        if not self.is_fitted or self.fitted_results is None:
            raise ModelNotFittedError("Model execution halted: Run fit routine before calling predict paths.")
        
        if steps <= 0:
            raise ValueError("Prediction horizon length steps must be a positive integer greater than 0.")

        forecast = self.fitted_results.forecast(steps=steps)
        return pd.Series(forecast, name="sarima_forecast")