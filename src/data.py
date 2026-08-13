"""Data ingestion and preprocessing module for time series workflows."""

from __future__ import annotations

import logging
import os
from .config import DataConfig
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class DataClass:
    """Handles data ingestion, validation, and train/test splitting for time series datasets."""

    def __init__(self, config: DataConfig) -> None:
        """Initializes DataClass and automatically triggers data ingestion.

        Args:
            config (DataConfig): Configuration parameters for the data layer.
        """
        self.config = config
        self._data: Optional[pd.DataFrame] = None
        self._train: Optional[pd.DataFrame] = None
        self._test: Optional[pd.DataFrame] = None

        self.load_data()

    @property
    def data(self) -> pd.DataFrame:
        """Gets the full dataset dataframe."""
        if self._data is None:
            raise ValueError("Dataset has not been loaded successfully.")
        return self._data

    @property
    def train(self) -> pd.DataFrame:
        """Gets the training dataset split."""
        if self._train is None:
            raise ValueError("Training split has not been initialized.")
        return self._train

    @property
    def test(self) -> pd.DataFrame:
        """Gets the testing dataset split."""
        if self._test is None:
            raise ValueError("Testing split has not been initialized.")
        return self._test

    def _read_pd(self, path: str) -> Optional[pd.DataFrame]:
        """Reads a pandas DataFrame from multiple supported file extensions.

        Args:
            path (str): File system path to parse.

        Returns:
            Optional[pd.DataFrame]: Loaded DataFrame or None if an unsupported format.
        """
        _, ext = os.path.splitext(path.lower())

        if ext == ".csv":
            return pd.read_csv(path)
        elif ext in [".xlsx", ".xls"]:
            return pd.read_excel(path)
        elif ext == ".parquet":
            return pd.read_parquet(path)

        logger.error(
            f"Unsupported format extension '{ext}' for file: {path}. "
            "Please ensure the file has a .csv, .xlsx, or .parquet extension."
        )
        return None

    def load_data(self) -> None:
        """Loads data from disk, configures temporal indexes, and creates data splits."""
        logger.info(f"Loading data from: {self.config.path}")
        try:
            df = self._read_pd(self.config.path)
            if df is None:
                raise ValueError(f"Failed to read data payload from {self.config.path}")

            # Handle Datetime Index conversion if specified
            if self.config.index_col:
                if self.config.index_col in df.columns:
                    df[self.config.index_col] = pd.to_datetime(
                        df[self.config.index_col]
                    )
                    df.set_index(self.config.index_col, inplace=True)
                    df.sort_index(inplace=True)
                else:
                    logger.warning(
                        f"Index column '{self.config.index_col}' not found in DataFrame columns."
                    )

            # Validate Index Structure for Downstream Time Series modeling
            if not isinstance(df.index, (pd.DatetimeIndex, pd.PeriodIndex)):
                logger.warning(
                    "DataFrame index is not a DatetimeIndex or PeriodIndex. "
                    "Downstream visualization tools may encounter sorting or parsing anomalies."
                )

            self._data = df
            self._train = df.iloc[: self.config.split_size, :]
            self._test = df.iloc[self.config.split_size :, :]

            assert self._train is not None and self._test is not None
            logger.info(
                f"Ingestion successful. Shapes -> Full: {self._data.shape}, "
                f"Train: {self._train.shape}, Test: {self._test.shape}"
            )

        except Exception as e:
            logger.error(
                f"Critical execution block fault while loading file: {self.config.path}. Error: {e}"
            )
            raise e
