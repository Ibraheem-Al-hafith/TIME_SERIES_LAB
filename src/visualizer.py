
"""Time series visualization module providing lifecycle-aware cached plotting tools.

This module contains the Visualizer class for orchestrating diagnostic, decomposition,
envelope, and seasonal plots for time series evaluation.
"""

from __future__ import annotations

import logging
import os
import shutil
from typing import Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # Ensure headless thread-safe execution
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import statsmodels.api as sm

from src.config import VisualizationConfig
from src.data import DataClass

logger = logging.getLogger(__name__)

class Visualizer:
    """Orchestrates visualization generations for time series evaluation, utilizing local caching."""

    def __init__(self, dataset: DataClass, config: VisualizationConfig) -> None:
        """Initializes the visualizer instance.

        Args:
            dataset: Connected dataset ingestor utility.
            config: Visual configuration parameters.
        """
        self._dataset = dataset
        self.config = config

        if hasattr(self.config, "style_theme") and self.config.style_theme in plt.style.available:
            plt.style.use(self.config.style_theme)
        else:
            plt.style.use("ggplot")

    @property
    def dataset(self) -> DataClass:
        """Gets the active dataset data utility context."""
        return self._dataset

    @dataset.setter
    def dataset(self, new_dataset: DataClass) -> None:
        """Binds a new dataset instance and updates the run_id to ensure cache isolation.

        Args:
            new_dataset: New DataClass instance to bind.
        """
        logger.info(
            "Dataset updated. Purging cached plots at %s to enforce cache isolation.",
            self.config.plot_path,
        )
        self._dataset = new_dataset
        if os.path.exists(self.config.plot_path):
            shutil.rmtree(self.config.plot_path)

    def _get_target_path(self, filename: str) -> str:
        """Builds an execution directory path based on the isolated run identifier.

        Args:
            filename: Name of the output image file.

        Returns:
            str: Absolute or relative output path.
        """
        target_dir = self.config.plot_path
        os.makedirs(target_dir, exist_ok=True)
        return os.path.join(target_dir, filename)

    def _handle_plot_lifecycle(self, filename: str) -> Tuple[bool, str]:
        """Checks for existing files in the cache to avoid redundant plotting operations.

        Args:
            filename: Target filename descriptor.

        Returns:
            Tuple[bool, str]: (is_cached, absolute_file_path_destination)
        """
        target_path = self._get_target_path(filename)
        #################################################################################
        #           TEMPORARY FIX, REMOVING THE OLD PLOT AND PLOT A NEW ONE
        #################################################################################

        if os.path.exists(target_path) and False: ################# THIS WILL NOT EXECUTE
            logger.info("Cache Hit: Displaying cached visualization: %s", target_path)
            return True, target_path 

        return False, target_path

    def _configure_time_axis(
        self,
        ax: plt.Axes,
        index: Union[pd.Index, pd.DatetimeIndex, pd.PeriodIndex],
        max_ticks: int = 15,
    ) -> None:
        """Configures X-axis ticks using series frequency and index boundaries.

        Ensures that all data points remain rendered in the line plot while limiting
        displayed tick labels to a clean, non-overlapping subset grounded strictly
        in the dataset's actual time observations.

        Args:
            ax: Target matplotlib axes instance.
            index: Dataset time index containing actual observation labels.
            max_ticks: Preferred maximum number of visible tick marks on the axis.
        """
        if len(index) == 0:
            return

        # Standardize PeriodIndex to DatetimeIndex where applicable
        if isinstance(index, pd.PeriodIndex):
            dt_index = index.to_timestamp()
        elif isinstance(index, pd.DatetimeIndex):
            dt_index = index
        else:
            # Attempt datetime conversion if string formatted
            try:
                dt_index = pd.to_datetime(index)
            except Exception:
                dt_index = None

        total_obs = len(index)

        # Case 1: Datetime Index - Frequency Aware Formatting
        if dt_index is not None and not dt_index.empty:
            inferred_freq = dt_index.inferred_freq or getattr(dt_index, "freqstr", None)

            if inferred_freq:
                freq_code = str(inferred_freq).split("-")[0].upper()
                if "D" in freq_code or "B" in freq_code:
                    interval = max(1, total_obs // max_ticks)
                    ax.xaxis.set_major_locator(mdates.DayLocator(interval=interval))
                    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
                elif "M" in freq_code or "MS" in freq_code:
                    interval = max(1, total_obs // max_ticks)
                    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=interval))
                    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
                elif "Q" in freq_code:
                    interval = max(1, total_obs // max_ticks)
                    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=interval * 3))
                    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-Q%q"))
                elif "A" in freq_code or "Y" in freq_code:
                    interval = max(1, total_obs // max_ticks)
                    ax.xaxis.set_major_locator(mdates.YearLocator(base=interval))
                    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
                else:
                    # Generic datetime fallback using observation step size
                    step = max(1, total_obs // max_ticks)
                    tick_indices = np.arange(0, total_obs, step)
                    ax.set_xticks(dt_index[tick_indices])
                    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            else:
                # Fallback step-based sampling on true index dates
                step = max(1, total_obs // max_ticks)
                tick_indices = np.arange(0, total_obs, step)
                ax.set_xticks(dt_index[tick_indices])
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

            ax.figure.autofmt_xdate(rotation=45, ha="right")

        # Case 2: Standard/Sequential Non-Datetime Index
        else:
            step = max(1, total_obs // max_ticks)
            tick_indices = np.arange(0, total_obs, step)
            ax.set_xticks(tick_indices)
            ax.set_xticklabels([str(index[i]) for i in tick_indices], rotation=45, ha="right")

    def plot_seasonal(
        self,
        target_col: str,
        period: int,
        show_mean: bool = True,
        cmap_name: str = "viridis",
    ) -> str:
        """Generates an overlaid seasonal cycle plot to inspect periodic behavior across time.

        Slices the target time series into `len(series) // period` consecutive lines,
        plotting each cycle over an normalized sub-period index (0 to period - 1) on
        a single axis. Includes a continuous colormap gradient and an optional average profile.

        Args:
            target_col: Target variable column name in the dataset.
            period: The seasonal cycle period length (e.g., 12 for monthly data, 7 for daily).
            show_mean: Whether to overlay an overall average seasonal trajectory.
            cmap_name: Matplotlib color map name used to color consecutive periods.

        Returns:
            str: Path to saved output image file.

        Raises:
            ValueError: If period is non-positive or if dataset size is insufficient.
            KeyError: If target_col is not found in the dataset.
        """
        if period <= 0:
            raise ValueError(f"Seasonal period must be a positive integer, got {period}.")

        filename = f"seasonal_plot_{target_col}_p{period}.png"
        is_cached, save_path = self._handle_plot_lifecycle(filename)
        if is_cached:
            return save_path

        if target_col not in self._dataset.data.columns:
            raise KeyError(f"Target column '{target_col}' not found in dataset columns.")

        series = self._dataset.data[target_col].dropna().sort_index()
        total_obs = len(series)
        num_cycles = total_obs // period

        if num_cycles < 1:
            raise ValueError(
                f"Dataset length ({total_obs}) is shorter than seasonal period ({period}). "
                "At least one complete cycle is required."
            )

        # Slice data into num_cycles complete sub-periods
        usable_len = num_cycles * period
        values = series.iloc[:usable_len].values.reshape(num_cycles, period)

        fig, ax = plt.subplots(figsize=self.config.default_figsize, dpi=self.config.dpi)
        x_axis = np.arange(period)

        # Generate color gradient for cycles
        cmap = plt.get_cmap(cmap_name)
        colors = [cmap(i / max(1, num_cycles - 1)) for i in range(num_cycles)]

        # Extract period labels for colorbar or legend context
        cycle_labels = []
        if isinstance(series.index, (pd.DatetimeIndex, pd.PeriodIndex)):
            dt_index = (
                series.index.to_timestamp()
                if isinstance(series.index, pd.PeriodIndex)
                else series.index
            )
            for i in range(num_cycles):
                start_dt = dt_index[i * period]
                cycle_labels.append(start_dt.strftime("%Y-%m-%d"))
        else:
            for i in range(num_cycles):
                cycle_labels.append(f"Cycle {i + 1}")

        # Plot individual period lines
        for i in range(num_cycles):
            ax.plot(
                x_axis,
                values[i, :],
                color=colors[i],
                alpha=0.6,
                linewidth=1.2,
                label=cycle_labels[i] if num_cycles <= 10 else None,
            )

        # Overlay mean seasonal pattern if requested
        if show_mean:
            mean_cycle = np.mean(values, axis=0)
            ax.plot(
                x_axis,
                mean_cycle,
                color="#e74c3c",
                linewidth=2.5,
                linestyle="--",
                label="Mean Seasonal Path",
            )

        # Format plot layout
        ax.set_title(
            f"Seasonal Cycle Analysis Map ({target_col.upper()})\n"
            f"Period Length = {period} | Total Cycles = {num_cycles}",
            fontsize=12,
            fontweight="bold",
            loc="left",
        )
        ax.set_xlabel("Seasonal Period Step (0 to Period-1)")
        ax.set_ylabel("Amplitude Value")
        ax.set_xticks(x_axis)
        ax.grid(True, alpha=0.3)

        # Configure Legend vs Colorbar depending on cycle density
        if num_cycles <= 10:
            ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0.0)
        else:
            sm_mappable = plt.cm.ScalarMappable(
                cmap=cmap,
                norm=mcolors.Normalize(vmin=1, vmax=num_cycles),
            )
            sm_mappable.set_array([])
            cbar = fig.colorbar(sm_mappable, ax=ax, pad=0.02)
            cbar.set_label("Seasonal Cycle Progression (Old → New)", fontsize=10)
            if show_mean:
                ax.legend(loc="upper left")

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight", dpi=self.config.dpi)
        plt.close(fig)

        return save_path

    def plot_envelope_components(
        self,
        target_col: str,
        distance: Optional[int] = None,
        prominence: Optional[float] = None,
    ) -> str:
        """Analyzes time series composition behavior via envelope geometry.

        Args:
            target_col: Target variable column name.
            distance: Peak detection distance threshold.
            prominence: Peak detection prominence threshold.

        Returns:
            str: Path to saved output image.
        """
        filename = f"envelope_analysis_{target_col}.png"
        is_cached, save_path = self._handle_plot_lifecycle(filename)
        if is_cached:
            return save_path

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
            f"Envelope Stability Tracking Map (Correlation Coefficient: {correlation:.4f})",
            fontsize=10,
            loc="left",
        )
        ax2.set_xlabel("Time Dimension Axis")
        ax2.set_ylabel("Absolute Distance")
        ax2.legend(loc="upper left")

        # Configure frequency-aware readable X-axis
        self._configure_time_axis(ax2, series.index)

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight", dpi=self.config.dpi)
        plt.close(fig)
        return save_path

    def plot_seasonal_decomposition(
        self, target_col: str, period: Optional[int] = None
    ) -> str:
        """Decomposes time series into Trend, Seasonal, and Residual components.

        Args:
            target_col: Target column name.
            period: Seasonal decomposition period length.

        Returns:
            str: Saved output image path.
        """
        filename = f"decomposition_{target_col}.png"
        is_cached, save_path = self._handle_plot_lifecycle(filename)
        if is_cached:
            return save_path

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

        # Apply readable frequency-aware tick layout to bottom subplot
        self._configure_time_axis(axes[-1], series.index)
        axes[-1].set_xlabel("Time Dimension Axis")

        plt.suptitle(
            f"Component Classical Decomposition Profile Analysis Framework ({target_col.upper()})",
            fontsize=13,
            fontweight="bold",
            y=0.99,
        )

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight", dpi=self.config.dpi)
        plt.close(fig)
        return save_path

    def plot_autocorrelation(
        self,
        target_col: str,
        series: Optional[pd.Series] = None,
        filename: Optional[str] = None,
    ) -> str:
        """Renders Autocorrelation (ACF) and Partial Autocorrelation (PACF) diagnostic charts.

        Args:
            target_col: Target column name.
            series: Optional time series override.
            filename: Optional output filename override.

        Returns:
            str: Path to saved output image file.
        """
        if filename is None:
            filename = f"autocorrelation_{target_col}.png"
        is_cached, save_path = self._handle_plot_lifecycle(filename)
        if is_cached:
            return save_path

        if series is None:
            series = self._dataset.data[target_col]
        series = series.dropna().sort_index()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5), dpi=self.config.dpi)

        sm.graphics.tsa.plot_acf(
            series,
            lags=self.config.acf_lags,
            ax=ax1,
            color="#1abc9c",
            vlines_kwargs={"color": "#16a085"},
        )
        ax1.set_title("Autocorrelation Function (ACF Map)", fontsize=11, fontweight="bold")
        ax1.set_xlabel("Lag Spacing Steps")
        ax1.set_ylabel("Correlation Weight Metric")

        sm.graphics.tsa.plot_pacf(
            series,
            lags=self.config.acf_lags,
            ax=ax2,
            color="#e67e22",
            vlines_kwargs={"color": "#d35400"},
        )
        ax2.set_title("Partial Autocorrelation Function (PACF Map)", fontsize=11, fontweight="bold")
        ax2.set_xlabel("Lag Spacing Steps")

        plt.suptitle(
            f"Memory Matrix Diagnostics Correlation Profiles ({target_col})",
            fontsize=13,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight", dpi=self.config.dpi)
        plt.close(fig)
        return save_path

    def plot_line_series(self, target_col: str) -> str:
        """Plots full sequential training and testing dataset slices over a unified timeline chart.

        Args:
            target_col: Column name of target metric.

        Returns:
            str: Path to saved output figure image.
        """
        filename = f"line_series_{target_col}.png"
        is_cached, save_path = self._handle_plot_lifecycle(filename)
        if is_cached:
            return save_path

        fig, ax = plt.subplots(figsize=self.config.default_figsize, dpi=self.config.dpi)

        # Plot full training and testing series continuously
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

        # Configure readable time axis across full combined range
        self._configure_time_axis(ax, self._dataset.data.index)

        ax.set_xlabel("Time Dimension Axis")
        ax.set_ylabel("Amplitude Values")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight", dpi=self.config.dpi)
        plt.close(fig)
        return save_path

    def plot_predictions_vs_actuals(
        self,
        predictions: Union[pd.Series, Dict[str, pd.Series]],
        target_col: str,
        filename: Optional[str] = None,
    ) -> str:
        """Plots out-of-sample inference curves against ground truth evaluation test data.

        Supports evaluating a single model prediction series or multiple candidate models 
        simultaneously via a model_name-to-series mapping.

        Args:
            predictions: Predicted values as a single Series or a dictionary mapping 
                model names to their respective prediction Series.
            target_col: Name of the target metric column within the test dataset.
            filename: Optional custom filename override. Defaults to a standard 
                naming pattern based on `target_col`.

        Returns:
            str: Saved figure image file path.

        Raises:
            KeyError: If target_col is absent from the test dataset.
            ValueError: If predictions dictionary is empty or contains no valid overlapping data.
        """
        if filename is None:
            filename = f"predictions_vs_actuals_{target_col}.png"

        is_cached, save_path = self._handle_plot_lifecycle(filename)
        if is_cached:
            return save_path

        if target_col not in self._dataset.test.columns:
            raise KeyError(
                f"Target column '{target_col}' not found in dataset test slice."
            )

        actuals = self._dataset.test[target_col].dropna()

        # Standardize predictions input to a dictionary structure: Dict[str, pd.Series]
        if isinstance(predictions, pd.Series):
            model_dict: Dict[str, pd.Series] = {"Model Prediction": predictions}
        elif isinstance(predictions, dict):
            if not predictions:
                raise ValueError("The 'predictions' dictionary cannot be empty.")
            model_dict = predictions
        else:
            raise TypeError(
                f"Expected 'predictions' to be pd.Series or Dict[str, pd.Series], "
                f"got {type(predictions).__name__}."
            )

        fig, ax = plt.subplots(
            figsize=self.config.default_figsize, dpi=self.config.dpi
        )

        # Plot Ground Truth Actuals
        ax.plot(
            actuals.index,
            actuals.values,
            label="Ground Truth Actuals",
            color="#2c3e50",
            linewidth=2.0,
            marker="o",
            markersize=3,
            alpha=0.9,
            zorder=3,
        )

        # Visual customization palettes for multi-model overlay
        colors = [
            "#e67e22", "#27ae60", "#2980b9", "#8e44ad", 
            "#d35400", "#16a085", "#c0392b", "#7f8c8d"
        ]
        markers = ["x", "^", "s", "D", "v", "*", "p", "+"]
        linestyles = ["--", "-.", ":", "-"]

        metrics_summary = []
        is_single_model = len(model_dict) == 1

        for idx, (model_name, pred_series) in enumerate(model_dict.items()):
            # Align actuals and model predictions on matching time indices
            aligned_df = pd.DataFrame(
                {"actuals": actuals, "predictions": pred_series}
            ).dropna()

            if aligned_df.empty:
                logger.warning(
                    "No overlapping data points between actuals and predictions for model: %s",
                    model_name,
                )
                continue

            # Compute error metrics
            err = aligned_df["actuals"] - aligned_df["predictions"]
            mae = float(np.mean(np.abs(err)))
            rmse = float(np.sqrt(np.mean(err**2)))
            metrics_summary.append((model_name, mae, rmse))

            color = colors[idx % len(colors)]
            marker = markers[idx % len(markers)]
            linestyle = linestyles[idx % len(linestyles)]

            # Construct legend label with inline metrics if multiple models are rendered
            label = (
                f"{model_name} (MAE: {mae:.2f}, RMSE: {rmse:.2f})"
                if not is_single_model
                else model_name
            )

            ax.plot(
                aligned_df.index,
                aligned_df["predictions"],
                label=label,
                color=color,
                linewidth=1.5,
                linestyle=linestyle,
                marker=marker,
                markersize=4,
                alpha=0.85,
                zorder=2,
            )

            # Draw shaded residual band only when visualizing a single model to avoid clutter
            if is_single_model:
                ax.fill_between(
                    aligned_df.index,
                    aligned_df["actuals"],
                    aligned_df["predictions"],
                    color="#e74c3c",
                    alpha=0.15,
                    label="Residual Error Deviances",
                    zorder=1,
                )

        # Configure dynamic header details
        if is_single_model and metrics_summary:
            _, mae, rmse = metrics_summary[0]
            title_text = (
                f"Model Prediction vs Actuals Evaluation Profiler ({target_col})\n"
                f"Mean Absolute Error: {mae:.4f} | Root Mean Squared Error: {rmse:.4f}"
            )
        else:
            title_text = (
                f"Multi-Model Comparative Inference Map ({target_col})\n"
                f"Evaluating {len(metrics_summary)} Model Candidate(s)"
            )

        ax.set_title(title_text, fontsize=11, fontweight="bold", loc="left")
        ax.set_xlabel("Time Dimension Axis")
        ax.set_ylabel("Amplitude Coordinate Target Levels")

        # Apply readable frequency-aware tick layout across the actuals index
        self._configure_time_axis(ax, actuals.index)

        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight", dpi=self.config.dpi)
        plt.close(fig)
        return save_path

    def plot_in_sample_fit(
        self,
        actual: pd.Series,
        fitted: pd.Series,
        title: str,
        filename: str,
        line_color: str = "blue",
    ) -> str:
        """Overlays in-sample fitted values against ground truth training data.

        Args:
            actual: Actual in-sample training series.
            fitted: Fitted model values series.
            title: Title text string for display on plot header.
            filename: Output path file name.
            line_color: Fitted series line color.

        Returns:
            str: Path to saved output image file.
        """
        save_path = self._get_target_path(filename)
        fig, ax = plt.subplots(figsize=(10, 4), dpi=self.config.dpi)

        ax.plot(
            actual.index,
            actual.values,
            label="Actual (In-Sample)",
            color="black",
            linewidth=1.5,
        )
        ax.plot(
            fitted.index,
            fitted.values,
            label="Fitted Path",
            color=line_color,
            linestyle="--",
        )
        ax.set_title(title, fontsize=11, fontweight="bold", loc="left")
        ax.legend(loc="upper left")
        ax.grid(True, linestyle=":", alpha=0.6)

        # Apply readable frequency-aware tick layout
        self._configure_time_axis(ax, actual.index)

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        return save_path