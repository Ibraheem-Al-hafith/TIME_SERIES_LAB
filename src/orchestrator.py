"""Unified execution orchestrator for automated time series experimentation.

This module provides the ExperimentOrchestrator which acts as the central execution
bridge coordinating data ingestion, dynamic model factory generation, performance
scoring portfolios, and isolated visualization storage.
"""

import logging
from typing import Any, Dict, Mapping, Optional, Type

import pandas as pd

# Import framework dependencies based on system blueprints
from src.config import Config
from src.data import DataClass
from src.metrics import ScoringEngine
from src.models import (
    BaseModel,
    DecompositionModel,
    ExponentialSmoothingModel,
    SARIMAModel,
)
from src.visualizer import Visualizer

# Setup module-level logger
logger = logging.getLogger(__name__)


# =====================================================================
# CUSTOM EXCEPTIONS DEFINITION
# =====================================================================

class OrchestrationError(Exception):
    """Base exception class for all errors inside the execution orchestrator."""
    pass


class TargetColumnNotFoundError(OrchestrationError):
    """Raised when the specified target data column is missing or misnamed."""
    pass


class InvalidSplitHorizonError(OrchestrationError):
    """Raised when the evaluation data split holds zero valid elements."""
    pass


class UnsupportedModelTypeError(OrchestrationError):
    """Raised when an unmapped or unrecognized model string tag is requested."""
    pass


# =====================================================================
# PIPELINE REGISTRY
# =====================================================================

# Static mapping binding string identifiers to concrete forecasting wrappers
MODEL_REGISTRY: Mapping[str, Type[BaseModel]] = {
    "decompose": DecompositionModel,
    "exponential_smoothing": ExponentialSmoothingModel,
    "sarima": SARIMAModel,
}


# =====================================================================
# CORE ORCHESTRATOR IMPLEMENTATION
# =====================================================================

class ExperimentOrchestrator:
    """Manages full lifecycle execution runs for time series forecasting pipelines.

    Coordinates data ingestion, short-lived stateless model instantiation, dynamic
    metric evaluation, and isolated visual asset caching.
    """

    def __init__(self, global_config: Config) -> None:
        """Initializes the orchestrator using a global immutable configuration layout.

        Args:
            global_config: A populated, validated system Config instance.
        """
        self.config: Config = global_config
        
        # Instantiate the data abstraction and scoring layers natively
        self.data_layer: DataClass = DataClass(self.config.data)
        self.scoring_engine: ScoringEngine = ScoringEngine(self.config.scoring)
        
        logger.debug("ExperimentOrchestrator successfully constructed and wired.")

    def run_single_model(self, model_type: str, target_column: str) -> Dict[str, Any]:
        """Runs an isolated training, forecasting, scoring, and visualization pipeline.

        Args:
            model_type: The lookup string key for the model (e.g., 'sarima').
            target_column: The target tracking column name to process.

        Returns:
            A dictionary tracking performance metrics, predictions, and runtime IDs.

        Raises:
            UnsupportedModelTypeError: If the string key does not match the registry.
            TargetColumnNotFoundError: If the target column is missing from data frames.
            InvalidSplitHorizonError: If the out-of-sample test horizon is empty.
        """
        logger.info("Initiating single model experiment run. Model: '%s'", model_type)

        # 1. Resolve concrete model factory allocations dynamically
        if model_type not in MODEL_REGISTRY:
            raise UnsupportedModelTypeError(
                f"Model identifier '{model_type}' is not recognized in the system pipeline registry. "
                f"Available models: {list(MODEL_REGISTRY.keys())}"
            )
        
        model_class: Type[BaseModel] = MODEL_REGISTRY[model_type]
        
        # Safely extract sub-configuration components using attribute mapping
        config_attr_map = {
            "decompose": "decompose",
            "exponential_smoothing": "exponential_smoothing",
            "sarima": "sarima"
        }
        
        config_block_name = config_attr_map[model_type]
        sub_config = getattr(self.config.models, config_block_name)

        # 2. Extract and validate training/testing slices securely
        train_df: pd.DataFrame = self.data_layer.train
        test_df: pd.DataFrame = self.data_layer.test

        if target_column not in train_df.columns or target_column not in test_df.columns:
            raise TargetColumnNotFoundError(
                f"Target tracking column '{target_column}' could not be located inside "
                f"the active dataset splits. Available columns: {list(train_df.columns)}"
            )

        forecast_steps: int = len(test_df)
        if forecast_steps <= 0:
            raise InvalidSplitHorizonError(
                "Evaluation split contains zero tracking elements. Verify split parameters "
                "or check testing allocation counts."
            )

        # 3. Instantiate and run short-lived stateless model wrappers
        model_instance: BaseModel = model_class(model_config=sub_config)
        
        logger.info("Executing training fit phase for model type '%s'...", model_type)
        model_instance.fit(df=train_df, target_col=target_column)
        
        logger.info("Generating out-of-sample predictions across %d steps...", forecast_steps)
        predictions: pd.Series = model_instance.predict(steps=forecast_steps)

        # 4. Score metrics calculation portfolios
        actuals: pd.Series = test_df[target_column]
        metrics_summary: Dict[str, float] = self.scoring_engine.evaluate(
            actuals=actuals, 
            predictions=predictions
        )

        # 5. Direct the Visualizer to isolate performance graphics outputs dynamically
        # Instantiating a new Visualizer instance updates the runtime execution ID,
        # ensuring each experiment run is correctly isolated in its own folder.
        visualizer = Visualizer(dataset=self.data_layer, config=self.config.visualizer)
        logger.info("Caching graphical validation outputs to target run directory ...")
        visualizer.plot_predictions_vs_actuals(predictions=predictions, target_col=target_column)

        return {
            "model_type": model_type,
            "metrics": metrics_summary,
            # "run_id": visualizer.run_id,
            "predictions": predictions,
        }

    def run_all_models(self, target_column: str) -> Dict[str, Any]:
        """Benchmarks all registered models sequentially against the target column.

        Provides exception isolation to ensure a single model failure does not terminate
        the entire evaluation batch sweep.

        Args:
            target_column: The target tracking column name to process.

        Returns:
            A comprehensive benchmark dictionary summarizing successful and failed model runs.
        """
        logger.info("Starting global registry benchmarking loop for column: '%s'", target_column)
        
        successful_runs: Dict[str, Any] = {}
        failed_runs: Dict[str, str] = {}

        for model_type in MODEL_REGISTRY.keys():
            try:
                run_result = self.run_single_model(model_type=model_type, target_column=target_column)
                successful_runs[model_type] = run_result
            except Exception as exc:
                # Capture anomalies cleanly without interrupting the overall loop execution path
                logger.error(
                    "Catastrophic failure encountered during model execution wrapper run '%s': %s",
                    model_type, str(exc), exc_info=True
                )
                failed_runs[model_type] = f"{type(exc).__name__}: {str(exc)}"

        logger.info(
            "Global registry sweep completed. Successful executions: %d/%d.",
            len(successful_runs), len(MODEL_REGISTRY)
        )

        return {
            "summary": {
                "total_attempted": len(MODEL_REGISTRY),
                "successful_count": len(successful_runs),
                "failed_count": len(failed_runs)
            },
            "results": successful_runs,
            "errors": failed_runs
        }