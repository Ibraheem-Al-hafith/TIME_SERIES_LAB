"""Unit tests for the data ingestion and split orchestration layer (data.py)."""

from __future__ import annotations

# import os
import pytest
import pandas as pd
import numpy as np

from src.data import DataConfig, DataClass


@pytest.fixture
def sample_csv_data(tmp_path) -> str:
    """Creates a temporary valid CSV file representing a time series dataset."""
    date_range = pd.date_range(start="2026-01-01", periods=10, freq="D")
    df = pd.DataFrame(
        {"Date": date_range, "value": np.arange(10, 20), "feature": np.random.randn(10)}
    )
    file_path = tmp_path / "ts_test_data.csv"
    df.to_csv(file_path, index=False)
    return str(file_path)


@pytest.fixture
def sample_excel_data(tmp_path) -> str:
    """Creates a temporary Excel file representing a time series dataset."""
    date_range = pd.date_range(start="2026-01-01", periods=5, freq="D")
    df = pd.DataFrame({"Date": date_range, "value": np.arange(10, 15)})
    file_path = tmp_path / "ts_test_data.xlsx"
    df.to_excel(file_path, index=False)
    return str(file_path)


def test_data_class_initialization_and_splitting(sample_csv_data):
    """Validates that data splits correctly segment based on split_size configuration."""
    config = DataConfig(path=sample_csv_data, split_size=7, index_col="Date")
    data_layer = DataClass(config)

    # Validate absolute properties
    assert isinstance(data_layer.data, pd.DataFrame)
    assert len(data_layer.data) == 10
    assert len(data_layer.train) == 7
    assert len(data_layer.test) == 3

    # Check that indices were correctly assigned and ordered
    assert isinstance(data_layer.data.index, pd.DatetimeIndex)
    assert data_layer.train.index[-1] < data_layer.test.index[0]


def test_data_class_unsupported_extension(tmp_path):
    """Ensures a ValueError is raised when handling an unexpected format extension."""
    invalid_file = tmp_path / "data.txt"
    invalid_file.write_text("Invalid text format payload content.")

    config = DataConfig(path=str(invalid_file), split_size=5)

    with pytest.raises(ValueError):
        DataClass(config)


def test_data_class_missing_file():
    """Validates exception propagation when pointing to a non-existent disk path."""
    config = DataConfig(path="non_existent_file_path_matrix.csv", split_size=5)

    with pytest.raises(FileNotFoundError):
        DataClass(config)


def test_data_class_missing_index_column_warning(sample_csv_data, caplog):
    """Ensures standard operation fallback with log alerts when the configured index column is missing."""
    config = DataConfig(
        path=sample_csv_data, split_size=6, index_col="NonExistentColumn"
    )
    data_layer = DataClass(config)

    # Ingestion continues, but standard sequential indexing takes over
    assert len(data_layer.data) == 10
    assert not isinstance(data_layer.data.index, pd.DatetimeIndex)
    assert "not found in DataFrame columns" in caplog.text
