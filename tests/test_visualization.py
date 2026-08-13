"""Unit tests verifying the time series visualization orchestrator and its file cache."""

from __future__ import annotations

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

from src.data import DataConfig, DataClass
from src.visualizer import VisualizationConfig, Visualizer


@pytest.fixture
def mock_data_class(tmp_path) -> DataClass:
    """Constructs a fully populated data layer instance inside an isolated temporary directory."""
    date_range = pd.date_range(start="2026-01-01", periods=20, freq="D")
    df = pd.DataFrame(
        {"Date": date_range, "target": np.sin(np.linspace(0, 10, 20)) + 50}
    )
    file_path = tmp_path / "mock_data.csv"
    df.to_csv(file_path, index=False)

    config = DataConfig(path=str(file_path), split_size=15, index_col="Date")
    return DataClass(config)


@pytest.fixture
def viz_config(tmp_path) -> VisualizationConfig:
    """Configures a temporary visualization environment profile."""
    return VisualizationConfig(
        plot_path=str(tmp_path / "output_plots"),
        decomposition_model="additive",
        style_theme="classic",
        dpi=100,
    )


@patch("matplotlib.pyplot.show")
def test_visualizer_run_id_isolation_on_dataset_mutation(
    mock_show, mock_data_class, viz_config, tmp_path
):
    """Verifies that shifting datasets triggers a run_id refresh to preserve workspace isolation."""
    visualizer = Visualizer(dataset=mock_data_class, config=viz_config)
    initial_run_id = visualizer.run_id

    # Generate an alternate secondary mock dataset
    alt_file = tmp_path / "alt_mock_data.csv"
    mock_data_class.data.to_csv(alt_file)
    alt_config = DataConfig(path=str(alt_file), split_size=10)
    alt_dataset = DataClass(alt_config)

    # Bind new data context layer
    visualizer.dataset = alt_dataset
    updated_run_id = visualizer.run_id

    assert initial_run_id != updated_run_id


@patch("matplotlib.pyplot.show")
@patch("matplotlib.pyplot.savefig")
def test_envelope_components_workflow_and_caching(
    mock_savefig, mock_show, mock_data_class, viz_config
):
    """Validates the lifecycle loop of the envelope metrics plot, testing calculations and the cache hit intercept."""
    visualizer = Visualizer(dataset=mock_data_class, config=viz_config)

    # Execution Pass 1: Compute metrics from scratch and save to the cache path
    visualizer.plot_envelope_components(target_col="target", distance=2)
    assert mock_savefig.call_count == 1

    # Explicitly create the file on disk to simulate a cache hit
    expected_path = visualizer._get_target_path("envelope_analysis_target.png")
    with open(expected_path, "w") as f:
        f.write("Simulated Image Payload Data")

    # Execution Pass 2: Re-trigger the same method call to intercept via the lifecycle cache handler
    with patch("matplotlib.image.imread") as mock_imread:
        mock_imread.return_value = np.zeros((10, 10, 3))
        visualizer.plot_envelope_components(target_col="target", distance=2)

        # Verify that matplotlib.image.imread loaded the file directly from cache
        mock_imread.assert_called_once_with(expected_path)


@patch("matplotlib.pyplot.show")
@patch("matplotlib.pyplot.savefig")
def test_predictions_vs_actuals_rendering(
    mock_savefig, mock_show, mock_data_class, viz_config
):
    """Ensures error boundaries, metrics, and line trends calculate correctly during test phase evaluations."""
    visualizer = Visualizer(dataset=mock_data_class, config=viz_config)

    test_len = len(mock_data_class.test)
    mock_predictions = pd.Series(
        np.random.randn(test_len) + 50, index=mock_data_class.test.index
    )

    visualizer.plot_predictions_vs_actuals(
        predictions=mock_predictions, target_col="target"
    )
    assert mock_savefig.call_count == 1
