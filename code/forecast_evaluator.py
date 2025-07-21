import os
import glob
import logging
import tempfile
from typing import Optional
import pandas as pd
import numpy as np
import toml
from sklearn.metrics import mean_absolute_error, mean_squared_error

class ForecastEvaluator:
    """
    A class to evaluate forecast results against validation and training datasets.
    """

    def __init__(
        self,
        forecast_folder: str,
        val_path: str,
        train_path: str,
        output_dir: Optional[str] = None
    ):
        """
        Initialize the evaluator with paths.

        :param forecast_folder: Folder containing forecast result files.
        :param val_path: Path to the validation dataset.
        :param train_path: Path to the training dataset.
        :param output_dir: Optional output directory for filtered training batches.
        """
        self.forecast_folder = forecast_folder
        self.val_path = val_path
        self.train_path = train_path
        self.temporary_dir = tempfile.TemporaryDirectory() if output_dir is None else None
        self.output_dir = output_dir or self.temporary_dir.name
        self.forecast_df = pd.DataFrame()
        self.val_df = pd.DataFrame()
        self.train_df = pd.DataFrame()
        self.metrics = {}
    def cleanup(self) -> None:
        """
        Clean up the temporary directory if one was created.
        """
        if self.temporary_dir:
            self.temporary_dir.cleanup()

    def load_data(self, forecast_pattern: str = "forecast_batch_mae_sliced_*.parquet") -> None:
        """
        Load forecast, validation, and training data.

        :param forecast_pattern: Glob pattern to find forecast files.
        """
        forecast_files = glob.glob(os.path.join(self.forecast_folder, forecast_pattern))
        logging.info(f"Found {len(forecast_files)} forecast files.")
        self.forecast_df = pd.concat([pd.read_parquet(f) for f in forecast_files], ignore_index=True)
        self.val_df = pd.read_parquet(self.val_path)
        self.train_df = pd.read_parquet(self.train_path)

    def _compute_metrics(self, y_true: pd.Series, y_pred: pd.Series, label: str) -> None:
        """
        Compute and print error metrics.

        :param y_true: True values.
        :param y_pred: Predicted values.
        :param label: Label to identify the dataset (e.g., 'Validation', 'Training').
        """
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mse = mean_squared_error(y_true, y_pred)

        print(f"{label} MAE: {mae:.4f}")
        print(f"{label} RMSE: {rmse:.4f}")
        print(f"{label} MSE: {mse:.4f}")
        # Store only MAE for TOML writing
        self.metrics[label.lower()] = mae

    def evaluate_validation(self) -> None:
        """
        Evaluate forecast accuracy against validation data.
        """
        merged = pd.merge(self.forecast_df, self.val_df, on=["unique_id", "ds"], how="inner")
        self._compute_metrics(merged["y_true"], merged["TimeGPT"], label="Validation")

    def evaluate_training(self, batch_size: int = 20000, steps: int = 92) -> None:
        """
        Evaluate forecast accuracy against the training data after filtering.

        :param batch_size: Number of series IDs per batch.
        :param steps: Number of most recent steps to retain per series.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        unique_ids = self.train_df['unique_id'].unique()

        for i in range(0, len(unique_ids), batch_size):
            batch_ids = unique_ids[i:i + batch_size]
            batch_df = self.train_df[self.train_df['unique_id'].isin(batch_ids)]
            filtered = batch_df.groupby("unique_id", group_keys=False).tail(steps)

            batch_file = os.path.join(self.output_dir, f"filtered_last{steps}_batch_{i // batch_size + 1}.parquet")
            filtered.to_parquet(batch_file, engine="pyarrow", index=False, compression="snappy")
            logging.info(f"Saved {filtered.shape[0]} rows to {batch_file}")

        filtered_files = glob.glob(os.path.join(self.output_dir, f"filtered_last{steps}_batch_*.parquet"))
        filtered_df = pd.concat([pd.read_parquet(f) for f in filtered_files], ignore_index=True)

        merged = pd.merge(self.forecast_df, filtered_df, on=["unique_id", "ds"], how="inner")
        y_col = "y_true" if "y_true" in merged.columns else "y"
        self._compute_metrics(merged[y_col], merged["TimeGPT"], label="Training")

    def update_errors_in_toml(self, path: str, solution_path:str, documentation_path:str, backup_path:str) -> None:
        """
        Update or create a TOML file with the latest training and validation errors.
    
        :param path: Path to the TOML file to update.
        """
        data = toml.load(path) if os.path.exists(path) else {}
        data.setdefault("errors", {})
        data["errors"]["train_error"] = self.metrics.get("training")
        data["errors"]["validation_error"] = self.metrics.get("validation")
        data["assignment"]["solution"] = solution_path
        data["assignment"]["solution_backup"] = backup_path
        data["assignment"]["documentation"] = documentation_path
        with open(path, "w") as f:
            toml.dump(data, f)
        logging.info(f"Updated errors in TOML file at {path}")
