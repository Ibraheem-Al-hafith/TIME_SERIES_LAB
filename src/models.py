"""Model abstractions and forecasting implementations using Dependency Injection and Configurations.

This module provides time-series forecasting model abstractions (Decomposition,
Exponential Smoothing, and SARIMA) that accept a centralized DataClass object and leverage
model-specific configuration dataclasses from config.py.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple, Union

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Assuming config.py and data.py reside in the same directory.
from .config import DecomposeConfig, ExponentialConfig, SARIMAConfig
from .data import DataClass


class ConfigurationError(Exception):
    """Raised when configuration requirements, model settings, or column bindings are invalid."""


class ModelNotFittedError(Exception):
    """Raised when prediction or evaluation is attempted prior to fitting."""


class BaseModel(ABC):
    """Abstract Base Class for all forecasting models.

    Defines the standardized interface for model fitting, prediction, and target
    resolution using Dependency Injection via DataClass and configuration schemas.

    Attributes:
        is_fitted (bool): Indicates whether model parameters have been estimated.
        target_col (Optional[str]): The resolved target column name used during fit.
    """

    def __init__(self) -> None:
        """Initialize the base forecasting model."""
        self.is_fitted: bool = False
        self.target_col: Optional[str] = None

    def _resolve_target_series(
        self, data_obj: DataClass, target_col: Optional[str] = None
    ) -> pd.Series:
        """Extract and validate the target time series vector from DataClass.

        Args:
            data_obj (DataClass): Data container holding config and train/test splits.
            target_col (Optional[str]): Explicit override for the target column name.

        Returns:
            pd.Series: Clean target series vector.

        Raises:
            ConfigurationError: If no target column is specified, if train data is missing,
                or if the target column does not exist in the training dataset.
        """
        if data_obj is None:
            raise ConfigurationError("DataClass instance ('data_obj') cannot be None.")

        # Step 1: Resolve target column identifier
        resolved_target = target_col or getattr(getattr(data_obj, "config", None), "target", None)
        if not resolved_target:
            raise ConfigurationError(
                "Target column is not specified explicitly and data_obj.config.target is undefined."
            )

        # Step 2: Validate training dataframe presence
        if not hasattr(data_obj, "train") or data_obj.train is None or data_obj.train.empty:
            raise ConfigurationError("Injected 'data_obj' does not contain a valid non-empty 'train' DataFrame.")

        # Step 3: Check column existence in training set
        if resolved_target not in data_obj.train.columns:
            raise ConfigurationError(
                f"Target column '{resolved_target}' not found in training dataset. "
                f"Available columns: {list(data_obj.train.columns)}"
            )

        self.target_col = resolved_target
        series = data_obj.train[resolved_target].dropna()

        if series.empty:
            raise ConfigurationError(f"Target column '{resolved_target}' contains no valid data after dropna().")

        return series

    @abstractmethod
    def fit(self, data_obj: DataClass, target_col: Optional[str] = None) -> BaseModel:
        """Fit the forecasting model to the dataset using configuration settings.

        Args:
            data_obj (DataClass): Data container instance containing dataset and config.
            target_col (Optional[str]): Optional column name override. Defaults to config.target.

        Returns:
            BaseModel: Self instance for method chaining.
        """
        pass

    @abstractmethod
    def predict(self, steps: int) -> pd.Series:
        """Generate out-of-sample forecasts.

        Args:
            steps (int): Number of future periods to forecast.

        Returns:
            pd.Series: Forecasted values.
        """
        pass


class DecompositionModel(BaseModel):
    """Classical Additive or Multiplicative Seasonal Decomposition Forecaster.

    Uses `statsmodels.tsa.seasonal.seasonal_decompose` to break down time series
    into trend, seasonal, and residual components based on `DecomposeConfig`.

    Attributes:
        config (Optional[DecomposeConfig]): Model configuration instance.
        trend_slope (float): Linear trend slope extracted from historical trend component.
        trend_intercept (float): Linear trend intercept extracted from historical trend component.
        seasonal_pattern (np.ndarray): Repeated seasonal factor pattern.
        last_index (int): Time index offset tracking end of training sequence.
    """

    def __init__(self, config: Optional[DecomposeConfig] = None) -> None:
        """Initialize DecompositionModel with optional DecomposeConfig.

        Args:
            config (Optional[DecomposeConfig]): Configuration instance. If None, resolves
                from data_obj.config at fit time.
        """
        super().__init__()
        self.config: Optional[DecomposeConfig] = config
        self.trend_slope: float = 0.0
        self.trend_intercept: float = 0.0
        self.seasonal_pattern: Optional[np.ndarray] = None
        self.last_index: int = 0

    def _resolve_config(self, data_obj: DataClass) -> DecomposeConfig:
        """Resolve model configuration from explicit initialization or injected DataClass.

        Args:
            data_obj (DataClass): Injected dataset container.

        Returns:
            DecomposeConfig: Resolved decomposition configuration.

        Raises:
            ConfigurationError: If no valid DecomposeConfig can be found.
        """
        if self.config is not None:
            return self.config

        data_config = getattr(data_obj, "config", None)
        model_config = getattr(data_config, "decompose", None)

        if isinstance(model_config, DecomposeConfig):
            return model_config

        # Default fallback if not defined in data_obj.config
        return DecomposeConfig()

    def fit(self, data_obj: DataClass, target_col: Optional[str] = None) -> DecompositionModel:
        """Fit classical decomposition model on target series using statsmodels.

        Args:
            data_obj (DataClass): Injected data container.
            target_col (Optional[str]): Optional target column override.

        Returns:
            DecompositionModel: The fitted model instance.

        Raises:
            ConfigurationError: If data length is insufficient or values violate model assumptions.
        """
        series = self._resolve_target_series(data_obj, target_col)
        self.config = self._resolve_config(data_obj)

        period = self.config.period
        model_type = self.config.model

        if len(series) < period * 2:
            raise ConfigurationError(
                f"Insufficient data points ({len(series)}) for seasonal decomposition with period ({period}). "
                f"At least {period * 2} points required."
            )

        if model_type == "multiplicative" and (series <= 0).any():
            raise ConfigurationError(
                "Multiplicative decomposition requires strictly positive time series values."
            )

        # Execute statsmodels classical decomposition
        decomposition = seasonal_decompose(
            series,
            model=model_type,
            period=period,
            extrapolate_trend="freq",
        )

        trend = decomposition.trend.dropna()
        seasonal = decomposition.seasonal

        # Fit OLS linear model on the extracted trend to project out-of-sample trend
        x_trend = np.arange(len(series))
        y_trend = trend.values
        slope, intercept = np.polyfit(x_trend, y_trend, 1)

        self.trend_slope = float(slope)
        self.trend_intercept = float(intercept)

        # Extract 1 full period of seasonal pattern
        raw_seasonal_cycle = seasonal.values[-period:]
        if model_type == "additive":
            self.seasonal_pattern = raw_seasonal_cycle - np.mean(raw_seasonal_cycle)
        else:  # multiplicative
            mean_val = np.mean(raw_seasonal_cycle)
            self.seasonal_pattern = raw_seasonal_cycle / (mean_val if mean_val != 0 else 1.0)

        self.last_index = len(series)
        self.is_fitted = True

        return self

    def predict(self, steps: int) -> pd.Series:
        """Generate forecasts by projecting linear trend and applying periodic seasonality.

        Args:
            steps (int): Horizon step count for projection.

        Returns:
            pd.Series: Forecasted values.

        Raises:
            ModelNotFittedError: If the model has not been fitted prior to prediction.
            ValueError: If step count is non-positive.
        """
        if not self.is_fitted or self.config is None or self.seasonal_pattern is None:
            raise ModelNotFittedError("DecompositionModel must be fitted before generating predictions.")
        if steps <= 0:
            raise ValueError("Prediction steps must be a positive integer.")

        period = self.config.period
        future_x = np.arange(self.last_index, self.last_index + steps)
        projected_trend = self.trend_intercept + self.trend_slope * future_x

        seasonal_factors = np.array([self.seasonal_pattern[i % period] for i in future_x])

        if self.config.model == "additive":
            forecast = projected_trend + seasonal_factors
        else:  # multiplicative
            forecast = projected_trend * seasonal_factors

        return pd.Series(forecast, name=f"{self.target_col}_forecast")


class ExponentialSmoothingModel(BaseModel):
    """Holt-Winters Exponential Smoothing Forecaster controlled via ExponentialConfig.

    Attributes:
        config (Optional[ExponentialConfig]): Model hyperparameter configuration object.
        fitted_model (Any): Fitted statsmodels ExponentialSmoothingResults object.
    """

    def __init__(self, config: Optional[ExponentialConfig] = None) -> None:
        """Initialize ExponentialSmoothingModel with optional ExponentialConfig.

        Args:
            config (Optional[ExponentialConfig]): Configuration schema instance.
        """
        super().__init__()
        self.config: Optional[ExponentialConfig] = config
        self.fitted_model: Any = None

    def _resolve_config(self, data_obj: DataClass) -> ExponentialConfig:
        """Resolve model configuration from explicit initialization or injected DataClass.

        Args:
            data_obj (DataClass): Injected dataset container.

        Returns:
            ExponentialConfig: Resolved configuration object.
        """
        if self.config is not None:
            return self.config

        data_config = getattr(data_obj, "config", None)
        model_config = getattr(data_config, "exponential", None)

        if isinstance(model_config, ExponentialConfig):
            return model_config

        return ExponentialConfig()

    def fit(self, data_obj: DataClass, target_col: Optional[str] = None) -> ExponentialSmoothingModel:
        """Fit Holt-Winters Exponential Smoothing estimator using resolved parameters.

        Args:
            data_obj (DataClass): Injected data container.
            target_col (Optional[str]): Optional target column override.

        Returns:
            ExponentialSmoothingModel: Fitted model instance.
        """
        series = self._resolve_target_series(data_obj, target_col)
        self.config = self._resolve_config(data_obj)

        model = ExponentialSmoothing(
            series,
            trend=self.config.trend,
            seasonal=self.config.seasonal,
            seasonal_periods=self.config.seasonal_periods,
            initialization_method="estimated",
        )
        self.fitted_model = model.fit()
        self.is_fitted = True

        return self

    def predict(self, steps: int) -> pd.Series:
        """Generate out-of-sample forecast using fitted Holt-Winters estimator.

        Args:
            steps (int): Horizon step count for projection.

        Returns:
            pd.Series: Forecasted values.
        """
        if not self.is_fitted or self.fitted_model is None:
            raise ModelNotFittedError("ExponentialSmoothingModel is not fitted.")
        if steps <= 0:
            raise ValueError("Steps must be a positive integer.")

        forecast = self.fitted_model.forecast(steps)
        forecast.name = f"{self.target_col}_forecast"
        return forecast


class SARIMAModel(BaseModel):
    """Seasonal AutoRegressive Integrated Moving Average Forecaster controlled via SARIMAConfig.

    Attributes:
        config (Optional[SARIMAConfig]): Model order parameter configuration object.
        fitted_model (Any): Fitted statsmodels SARIMAXResults object.
    """

    def __init__(self, config: Optional[SARIMAConfig] = None) -> None:
        """Initialize SARIMA model with optional SARIMAConfig.

        Args:
            config (Optional[SARIMAConfig]): Model order parameters.
        """
        super().__init__()
        self.config: Optional[SARIMAConfig] = config
        self.fitted_model: Any = None

    def _resolve_config(self, data_obj: DataClass) -> SARIMAConfig:
        """Resolve model configuration from explicit initialization or injected DataClass.

        Args:
            data_obj (DataClass): Injected dataset container.

        Returns:
            SARIMAConfig: Resolved configuration object.
        """
        if self.config is not None:
            return self.config

        data_config = getattr(data_obj, "config", None)
        model_config = getattr(data_config, "sarima", None)

        if isinstance(model_config, SARIMAConfig):
            return model_config

        return SARIMAConfig()

    def fit(self, data_obj: DataClass, target_col: Optional[str] = None) -> SARIMAModel:
        """Estimate SARIMAX parameters using configuration state.

        Args:
            data_obj (DataClass): Injected data container.
            target_col (Optional[str]): Optional target column override.

        Returns:
            SARIMAModel: Fitted model instance.
        """
        series = self._resolve_target_series(data_obj, target_col)
        self.config = self._resolve_config(data_obj)

        order: Tuple[int, int, int] = (self.config.p, self.config.d, self.config.q)
        seasonal_order: Tuple[int, int, int, int] = (
            self.config.P,
            self.config.D,
            self.config.Q,
            self.config.s if self.config.s is not None else 0,
        )

        model = SARIMAX(
            series,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self.fitted_model = model.fit(disp=False)
        self.is_fitted = True

        return self

    def predict(self, steps: int) -> pd.Series:
        """Project out-of-sample target predictions.

        Args:
            steps (int): Number of steps ahead to forecast.

        Returns:
            pd.Series: Forecast sequence.
        """
        if not self.is_fitted or self.fitted_model is None:
            raise ModelNotFittedError("SARIMAModel must be fitted before generating predictions.")
        if steps <= 0:
            raise ValueError("Steps must be a positive integer.")

        forecast = self.fitted_model.forecast(steps=steps)
        forecast.name = f"{self.target_col}_forecast"
        return forecast