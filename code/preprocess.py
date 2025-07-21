import numpy as np
import pandas as pd
import os
import traceback
import logging
from typing import Literal, Optional, Tuple
from load_export_data import KNDVIExtractor, get_default_paths

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class TimeGPTDataProcessor:
    def __init__(self, train: str, val: str, test: str, results: str, output_dir: str):
        self.extractor = KNDVIExtractor(train, val, test, results)
        self.output_traindata = os.path.join(output_dir, 'train_timegpt.parquet')
        self.output_traindataerrors = os.path.join(output_dir, 'trainerrors_timegpt.parquet')
        self.output_val = os.path.join(output_dir, 'val_timegpt.parquet')
        self.train_kndvi: Optional[np.ndarray] = None
        self.val_kndvi: Optional[np.ndarray] = None

    def load_data(self) -> None:
        try:
            train_data, val_data = self.extractor.load_stored_data()
            self.train_kndvi = self.convert_type_format(train_data)
            self.val_kndvi = self.convert_type_format(val_data)
            logging.info(f"Loaded and converted data. Train dtype: {self.train_kndvi.dtype}, Val dtype: {self.val_kndvi.dtype}")
        except Exception:
            logging.error("An error occurred while loading and converting data")
            logging.error(traceback.format_exc())

    @staticmethod
    def convert_type_format(dataset: np.ndarray, target_variable: str = 'kndvi') -> np.ndarray:
        if target_variable not in dataset:
            raise KeyError(f"Target variable '{target_variable}' not found in dataset")
        return dataset[target_variable].astype(np.float32)

    @staticmethod
    def convert_ndarray(datacube: np.ndarray, start_date: str, freq: str,
                        n_steps: Optional[int] = None, value_col: Literal["y", "y_true"] = "y") -> pd.DataFrame:
        assert datacube.ndim == 2, "Input must be a 2D array"

        n_samples, n_timesteps = datacube.shape
        if n_steps:
            assert n_steps <= n_timesteps, "n_steps cannot be greater than time dimension"
            time_index = pd.date_range(start=start_date, periods=n_steps, freq=freq)
        else:
            n_steps = n_timesteps
            time_index = pd.date_range(start=start_date, periods=n_steps, freq=freq)

        flat_values = datacube[:, -n_steps:].flatten()
        time_repeated = np.tile(time_index, n_samples)
        ids_repeated = np.repeat(np.arange(n_samples), len(time_index))

        return pd.DataFrame({
            "unique_id": ids_repeated,
            "ds": time_repeated,
            value_col: flat_values
        })

    def save_to_parquet(self) -> None:
        if self.train_kndvi is None or self.val_kndvi is None:
            raise ValueError("Data not loaded. Call load_data() before save_to_parquet().")

        full_time_index = pd.date_range("2000-03-01", periods=1004, freq="8D")

        train_df = self.convert_ndarray(
            datacube=self.train_kndvi[:, :-92],
            start_date="2000-03-01",
            freq="8D",
            value_col="y"
        )
        val_df = self.convert_ndarray(
            datacube=self.train_kndvi[:, -92:],
            start_date=str(full_time_index[912].date()),
            freq="8D",
            n_steps=92,
            value_col="y_true"
        )
        
        train_errors_df = self.convert_ndarray(
            datacube=self.train_kndvi[:, :],
            start_date="2000-03-01",
            freq="8D",
            value_col="y"
        )

        train_df.to_parquet(self.output_traindata, index=False)
        val_df.to_parquet(self.output_val, index=False)
        train_errors_df.to_parquet(self.output_traindataerrors, index=False)
        logging.info("Train and validation data exported to Parquet format.")
