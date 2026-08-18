"""Unified execution orchestrator for automated time series experimentation.

Coordinates data ingestion through `DataClass`, dynamic model instantiation via dependency
injection, evaluation scoring through `ScoringEngine`, and visual output generation.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Mapping, Optional, Type

import pandas as pd

from .config import Config
from .data import DataClass
from .metrics import ScoringEngine
from .models import (
    BaseModel,
    DecompositionModel,
    ExponentialSmoothingModel,
    SARIMAModel,
)
from .visualizer import Visualizer

logger = logging.getLogger(__name__)


# =====================================================================
# CUSTOM EXCEPTIONS DEFINITION
# =====================================================================

class OrchestrationError(Exception):
    """Base exception class for execution orchestrator errors."""


class TargetColumnNotFoundError(OrchestrationError):
    """Raised when the specified target data column is missing or misnamed."""


class InvalidSplitHorizonError(OrchestrationError):
    """Raised when the evaluation data split holds zero valid elements."""


class UnsupportedModelTypeError(OrchestrationError):
    """Raised when an unmapped or unrecognized model identifier key is requested."""


# =====================================================================
# PIPELINE REGISTRY
# =====================================================================

MODEL_REGISTRY: Mapping[str, Type[BaseModel]] = {
    "decompose": DecompositionModel,
    "exponential_smoothing": ExponentialSmoothingModel,
    "sarima": SARIMAModel,
}


# =====================================================================
# CORE ORCHESTRATOR IMPLEMENTATION
# =====================================================================

class ExperimentOrchestrator:
    """Manages execution runs for time series forecasting models.

    Uses `DataClass` as a centralized data layer and injects it into model instances.
    Coordinates fit, forecast, scoring, and visualization tasks.
    """

    def __init__(self, global_config: Config) -> None:
        """Initialize orchestrator with immutable global configuration.

        Args:
            global_config: System configuration layout.
        """
        self.config: Config = global_config
        self.data_layer: DataClass = DataClass(self.config.data)
        self.scoring_engine: ScoringEngine = ScoringEngine(self.config.scoring)

        logger.debug("ExperimentOrchestrator constructed and wired.")

    def run_single_model(
        self, model_type: str, target_column: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run an isolated training, forecasting, scoring, and visualization pipeline.

        Args:
            model_type: Key identifier for the model (e.g., 'sarima').
            target_column: Target column override name. If None, uses config default.

        Returns:
            Dictionary tracking metrics, forecast predictions, and model metadata.

        Raises:
            UnsupportedModelTypeError: If model_type key is unregistered.
            TargetColumnNotFoundError: If target column is missing from training dataset.
            InvalidSplitHorizonError: If evaluation horizon length is zero.
        """
        logger.info("Initiating single model experiment. Type: '%s'", model_type)

        if model_type not in MODEL_REGISTRY:
            raise UnsupportedModelTypeError(
                f"Model identifier '{model_type}' is unrecognized. "
                f"Available models: {list(MODEL_REGISTRY.keys())}"
            )

        model_class: Type[BaseModel] = MODEL_REGISTRY[model_type]

        # Resolve target column identifier
        resolved_target = (
            target_column
            or getattr(self.config.data, "target", None)
            or getattr(self.data_layer.config, "target", None)
        )

        if not resolved_target:
            raise TargetColumnNotFoundError(
                "Target column is not specified explicitly and configuration target is missing."
            )

        # Validate target column in training set via DataClass API
        train_df = self.data_layer.train
        test_df = self.data_layer.test

        if resolved_target not in train_df.columns:
            raise TargetColumnNotFoundError(
                f"Target column '{resolved_target}' not found in training dataset. "
                f"Available columns: {list(train_df.columns)}"
            )

        forecast_steps = len(test_df)
        if forecast_steps <= 0:
            raise InvalidSplitHorizonError(
                "Evaluation test split contains zero elements. Check dataset split parameters."
            )

        # Instantiate model wrapper and fit via DataClass dependency injection
        model_instance: BaseModel = model_class()

        logger.info("Executing fit phase for model '%s' using target '%s'...", model_type, resolved_target)
        model_instance.fit(data_obj=self.data_layer, target_col=resolved_target)

        logger.info("Generating predictions across %d steps horizon...", forecast_steps)
        predictions: pd.Series = model_instance.predict(steps=forecast_steps)

        # Retrieve actual ground truth values directly from test split via DataClass
        actuals: pd.Series = test_df[resolved_target]

        # --- FIX: Ensure predictions inherit the exact datetime index from test set ---
        predictions.index = actuals.index

        # Compute evaluation metrics
        metrics_summary: Dict[str, float] = self.scoring_engine.evaluate(
            actuals=actuals, predictions=predictions
        )

        # Generate output visual assets
        visualizer = Visualizer(dataset=self.data_layer, config=self.config.visualizer)
        logger.info("Caching graphical outputs to visualization directory...")
        visualizer.plot_predictions_vs_actuals(
            predictions=predictions, target_col=resolved_target
        )

        return {
            "model_type": model_type,
            "target_column": resolved_target,
            "metrics": metrics_summary,
            "predictions": predictions,
        }

    def run_all_models(
        self, target_column: Optional[str] = None
    ) -> Dict[str, Any]:
        """Benchmark all registered models against the target column.

        Provides exception isolation to ensure a single model failure does not break
        the full execution batch sweep.

        Args:
            target_column: Optional target column override name.

        Returns:
            Dictionary summarizing successful and failed model runs.
        """
        logger.info("Starting global benchmark sweep across registered models...")

        successful_runs: Dict[str, Any] = {}
        failed_runs: Dict[str, str] = {}

        for model_type in MODEL_REGISTRY:
            try:
                run_result = self.run_single_model(
                    model_type=model_type, target_column=target_column
                )
                successful_runs[model_type] = run_result
            except Exception as exc:
                logger.error(
                    "Failure encountered during model run '%s': %s",
                    model_type,
                    exc,
                    exc_info=True,
                )
                failed_runs[model_type] = f"{type(exc).__name__}: {exc}"

        logger.info(
            "Global registry sweep completed. Successful: %d/%d.",
            len(successful_runs),
            len(MODEL_REGISTRY),
        )

        return {
            "summary": {
                "total_attempted": len(MODEL_REGISTRY),
                "successful_count": len(successful_runs),
                "failed_count": len(failed_runs),
            },
            "results": successful_runs,
            "errors": failed_runs,
        }