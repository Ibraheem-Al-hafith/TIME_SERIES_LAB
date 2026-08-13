"""Advanced visualization orchestration suite with disk caching and runtime isolation features."""

from __future__ import annotations

import logging
import os
import uuid

# from dataclasses import dataclass
from typing import Optional, Tuple

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import statsmodels.api as sm
from .config import VisualizationConfig
from .data import DataClass

logger = logging.getLogger(__name__)


class Visualizer:
    """Orchestrates visualization generations for time series evaluation, utilizing local caching."""

    def __init__(self, dataset: DataClass, config: VisualizationConfig) -> None:
        """Initializes the visualizer instance.

        Args:
            dataset (DataClass): Connected dataset ingestor utility.
            config (VisualizationConfig): Visual configuration parameters.
        """
        self._dataset = dataset
        self.config = config
        self._run_id: str = self._generate_run_id()

        if self.config.style_theme in plt.style.available:
            plt.style.use(self.config.style_theme)
        else:
            plt.style.use("ggplot")

    @property
    def dataset(self) -> DataClass:
        """Gets the active dataset data utility context."""
        return self._dataset

    @dataset.setter
    def dataset(self, new_dataset: DataClass) -> None:
        """Binds a new dataset instance and updates the run_id to ensure cache isolation."""
        logger.info(
            "Dataset updated. Updating dynamic run_id to enforce cache isolation."
        )
        self._dataset = new_dataset
        self._run_id = self._generate_run_id()

    @property
    def run_id(self) -> str:
        """Gets the current execution sequence run_id string token."""
        return self._run_id

    def _generate_run_id(self) -> str:
        """Generates a brief unique random string token."""
        return uuid.uuid4().hex[:8]

    def _get_target_path(self, filename: str) -> str:
        """Builds an execution directory path based on the isolated run identifier."""
        target_dir = os.path.join(self.config.plot_path, f"run_{self._run_id}")
        os.makedirs(target_dir, exist_ok=True)
        return os.path.join(target_dir, filename)

    def _handle_plot_lifecycle(self, filename: str) -> Tuple[bool, str]:
        """Checks for existing files in the cache to avoid redundant plotting operations.

        Args:
            filename (str): Target filename descriptor.

        Returns:
            Tuple[bool, str]: (is_cached, absolute_file_path_destination)
        """
        target_path = self._get_target_path(filename)

        if os.path.exists(target_path):
            logger.info(f"Cache Hit: Displaying cached visualization: {target_path}")
            img = mpimg.imread(target_path)
            fig, ax = plt.subplots(
                figsize=self.config.default_figsize, dpi=self.config.dpi
            )
            ax.imshow(img)
            ax.axis("off")
            plt.tight_layout()
            plt.show()
            return True, target_path

        return False, target_path

    def plot_envelope_components(
        self,
        target_col: str,
        distance: Optional[int] = None,
        prominence: Optional[float] = None,
    ) -> None:
        """Analyzes time series composition behavior (additive vs multiplicative) via envelope geometry."""
        filename = f"envelope_analysis_{target_col}.png"
        is_cached, save_path = self._handle_plot_lifecycle(filename)
        if is_cached:
            return

        series = self._dataset.data[target_col].dropna().sort_index()
        y = series.values
        x = np.arange(len(y))
        dates = (
            series.index.to_timestamp()
            if isinstance(series.index, pd.PeriodIndex)
            else series.index
        )

        upper_peaks, _ = find_peaks(y, distance=distance, prominence=prominence)
        lower_peaks, _ = find_peaks(-y, distance=distance, prominence=prominence)

        u_indices = np.unique(np.concatenate(([0], upper_peaks, [len(y) - 1])))
        l_indices = np.unique(np.concatenate(([0], lower_peaks, [len(y) - 1])))

        upper_envelope = np.interp(x, u_indices, y[u_indices])
        lower_envelope = np.interp(x, l_indices, y[l_indices])

        mid_trend = (upper_envelope + lower_envelope) / 2
        envelope_spread = upper_envelope - lower_envelope
        correlation = pd.Series(mid_trend).corr(pd.Series(envelope_spread))

        if correlation > 0.5:
            verdict, explanation = (
                "MULTIPLICATIVE",
                "Spread widens proportionally with trend level.",
            )
        elif correlation < -0.5:
            verdict, explanation = (
                "INVERSE MULTIPLICATIVE",
                "Spread contracts as trend level increases.",
            )
        else:
            verdict, explanation = (
                "ADDITIVE",
                "Spread stays stable and independent of trend alterations.",
            )

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(14, 9), dpi=self.config.dpi, sharex=True
        )

        ax1.plot(
            dates,
            y,
            label=f"Raw Data ({target_col})",
            color="#2c3e50",
            alpha=0.6,
            linewidth=1.2,
        )
        ax1.plot(
            dates,
            upper_envelope,
            label="Upper Envelope",
            color="#27ae60",
            linestyle="--",
            alpha=0.8,
        )
        ax1.plot(
            dates,
            lower_envelope,
            label="Lower Envelope",
            color="#c0392b",
            linestyle="--",
            alpha=0.8,
        )
        ax1.plot(
            dates,
            mid_trend,
            label="Mid-Trend Baseline",
            color="#2980b9",
            linestyle="-.",
            linewidth=1.5,
        )
        ax1.set_title(
            f"Envelope Geometry Analysis\nVerdict: {verdict} ({explanation})",
            fontsize=12,
            fontweight="bold",
            loc="left",
        )
        ax1.set_ylabel("Amplitude Value")
        ax1.legend(loc="upper left")

        ax2.plot(
            dates,
            envelope_spread,
            label="Envelope Absolute Distance Width",
            color="#8e44ad",
            linewidth=1.5,
        )
        ax2.set_title(
            f"Envelope Stability Tracking Map (Correlation Coefficient Vector: {correlation:.4f})",
            fontsize=10,
            loc="left",
        )
        ax2.set_xlabel("Time Dimension Axis")
        ax2.set_ylabel("Absolute Distance")
        ax2.legend(loc="upper left")

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight", dpi=self.config.dpi)
        plt.show()

    def plot_seasonal_decomposition(
        self, target_col: str, period: Optional[int] = None
    ) -> None:
        """Decomposes the time series into Trend, Seasonal, and Residual attributes."""
        filename = f"decomposition_{target_col}.png"
        is_cached, save_path = self._handle_plot_lifecycle(filename)
        if is_cached:
            return

        series = self._dataset.data[target_col].dropna().sort_index()
        decomposition = sm.tsa.seasonal_decompose(
            series, model=self.config.decomposition_model, period=period
        )

        fig, axes = plt.subplots(
            4, 1, figsize=(14, 10), dpi=self.config.dpi, sharex=True
        )
        components = [
            (decomposition.observed, "Observed Historical Signal", "#2c3e50"),
            (decomposition.trend, "Extracted Trend Component", "#2980b9"),
            (
                decomposition.seasonal,
                f"Seasonal Waves ({self.config.decomposition_model.capitalize()})",
                "#27ae60",
            ),
            (decomposition.resid, "Residual Unstructured Noise Component", "#e74c3c"),
        ]

        for ax, (data, title, color) in zip(axes, components):
            ax.plot(data.index, data.values, color=color, linewidth=1.2)
            ax.set_title(title, fontsize=11, fontweight="bold", loc="left")
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Time Dimension Axis")
        plt.suptitle(
            f"Component Classical Decomposition Profile Analysis Framework ({target_col.upper()})",
            fontsize=13,
            fontweight="bold",
            y=0.99,
        )

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight", dpi=self.config.dpi)
        plt.show()

    def plot_autocorrelation(self, target_col: str) -> None:
        """Renders Autocorrelation (ACF) and Partial Autocorrelation (PACF) plots side-by-side."""
        filename = f"autocorrelation_{target_col}.png"
        is_cached, save_path = self._handle_plot_lifecycle(filename)
        if is_cached:
            return

        series = self._dataset.data[target_col].dropna().sort_index()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5), dpi=self.config.dpi)

        sm.graphics.tsa.plot_acf(
            series,
            lags=self.config.acf_lags,
            ax=ax1,
            color="#1abc9c",
            vlines_kwargs={"color": "#16a085"},
            # vlines_color="#16a085",
        )
        ax1.set_title(
            "Autocorrelation Function (ACF Map)", fontsize=11, fontweight="bold"
        )
        ax1.set_xlabel("Lag Spacing Steps")
        ax1.set_ylabel("Correlation Weight Metric")

        sm.graphics.tsa.plot_pacf(
            series,
            lags=self.config.acf_lags,
            ax=ax2,
            color="#e67e22",
            vlines_kwargs={"color": "#d35400"},
            # vlines_color="#d35400",
        )
        ax2.set_title(
            "Partial Autocorrelation Function (PACF Map)",
            fontsize=11,
            fontweight="bold",
        )
        ax2.set_xlabel("Lag Spacing Steps")

        plt.suptitle(
            f"Memory Matrix Diagnostics Correlation Profiles ({target_col})",
            fontsize=13,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight", dpi=self.config.dpi)
        plt.show()

    def plot_line_series(self, target_col: str) -> None:
        """Plots the full sequential train and test dataset slices over a unified timeline chart."""
        filename = f"line_series_{target_col}.png"
        is_cached, save_path = self._handle_plot_lifecycle(filename)
        if is_cached:
            return

        fig, ax = plt.subplots(figsize=self.config.default_figsize, dpi=self.config.dpi)

        ax.plot(
            self._dataset.train.index,
            self._dataset.train[target_col],
            label="Training Validation Slice",
            color="#34495e",
            linewidth=1.5,
        )
        ax.plot(
            self._dataset.test.index,
            self._dataset.test[target_col],
            label="Testing Validation Slice",
            color="#e74c3c",
            linewidth=1.5,
        )

        ax.set_title(
            f"Time Series Sequential Data Segments Map ({target_col})",
            fontsize=12,
            fontweight="bold",
            loc="left",
        )
        ax.set_xlabel("Time Dimension Axis")
        ax.set_ylabel("Amplitude Values")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight", dpi=self.config.dpi)
        plt.show()

    def plot_predictions_vs_actuals(
        self, predictions: pd.Series, target_col: str
    ) -> None:
        """Plots out-of-sample inference curves directly against real target arrays."""
        filename = f"predictions_vs_actuals_{target_col}.png"
        is_cached, save_path = self._handle_plot_lifecycle(filename)
        if is_cached:
            return

        actuals = self._dataset.test[target_col]
        aligned_df = pd.DataFrame(
            {"actuals": actuals, "predictions": predictions}
        ).dropna()

        fig, ax = plt.subplots(figsize=self.config.default_figsize, dpi=self.config.dpi)

        ax.plot(
            aligned_df.index,
            aligned_df["actuals"],
            label="Ground Truth Evaluation Benchmarks",
            color="#2c3e50",
            linewidth=1.5,
            marker="o",
            markersize=3,
            alpha=0.8,
        )
        ax.plot(
            aligned_df.index,
            aligned_df["predictions"],
            label="Model Generated Inference Matrix Projections",
            color="#e67e22",
            linewidth=1.5,
            linestyle="--",
            marker="x",
            markersize=4,
        )

        ax.fill_between(
            aligned_df.index,
            aligned_df["actuals"],
            aligned_df["predictions"],
            color="#e74c3c",
            alpha=0.15,
            label="Residual Error Deviances",
        )

        mae = np.mean(np.abs(aligned_df["actuals"] - aligned_df["predictions"]))
        rmse = np.sqrt(
            np.mean((aligned_df["actuals"] - aligned_df["predictions"]) ** 2)
        )

        ax.set_title(
            f"Model Predictions vs Actuals Evaluation Profiler ({target_col})\nMean Absolute Error: {mae:.4f} | Root Mean Squared Error: {rmse:.4f}",
            fontsize=11,
            fontweight="bold",
            loc="left",
        )
        ax.set_xlabel("Time Dimension Axis")
        ax.set_ylabel("Amplitude Coordinate Target Levels")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight", dpi=self.config.dpi)
        plt.show()
