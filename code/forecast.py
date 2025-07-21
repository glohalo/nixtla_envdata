import os
import logging
import pandas as pd
from dotenv import dotenv_values
from nixtla import NixtlaClient
from typing import Dict, List
import tempfile
import shutil

class TimeGPTForecaster:
    def __init__(self, config_path: str, data_dir: str, output_dir: str):
        self.config_path = config_path
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.config = self.load_config()
        self.client = self.init_client()
        self.date_features = ['month', 'quarter', 'year', 'dayofyear']
        self.data = self.load_data()

    def load_config(self) -> Dict[str, any]:
        """Load and parse configuration from a .env file."""
        config = dotenv_values(self.config_path)
        return {
            "api_key": config["API_KEY"],
            "forecast_horizon": int(config["FORECAST_HORIZONT"]),
            "confidence_interval": [int(config["FORECAST_HORIZONT"])],
            "batch_size": int(config["BATCH_SIZE"]),
            "num_partitions": int(config["NUM_PARTITIONS"])
        }

    def init_client(self) -> NixtlaClient:
        """Initialize and validate the Nixtla client."""
        client = NixtlaClient(api_key=self.config["api_key"])
        client.validate_api_key()
        return client

    def load_data(self) -> pd.DataFrame:
        """Load training data from the parquet file."""
        path = os.path.join(self.data_dir, 'train_timegpt.parquet')
        logging.info(f"Loading training data from {path}")
        return pd.read_parquet(path)


    def forecast_all_batches(self) -> None:
        """Generate forecasts for all batches and save them to disk."""
        unique_ids = self.data['unique_id'].unique().tolist()
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(0, len(unique_ids), self.config["batch_size"]):
                batch_ids = unique_ids[i:i + self.config["batch_size"]]
                logging.info(f"Processing batch {i // self.config['batch_size'] + 1} with {len(batch_ids)} IDs")
                batch_df = self.data[self.data['unique_id'].isin(batch_ids)]
    
                forecasts = self.client.forecast(
                    df=batch_df,
                    target_col='y',
                    h=self.config["forecast_horizon"],
                    level=self.config["confidence_interval"],
                    finetune_steps=20,
                    finetune_depth=5,
                    finetune_loss='mse',
                    model='timegpt-1-long-horizon',
                    date_features=self.date_features,
                    num_partitions=self.config["num_partitions"]
                )
    
                temp_path = os.path.join(tmpdir, f'forecast_batch_mae_sliced_{i // self.config["batch_size"] + 1}.parquet')
                forecasts.to_parquet(temp_path)
                shutil.copy(temp_path, self.output_dir)  # copy to final location if needed
                logging.info(f"Saved forecasts to {temp_path}")

def setup_logging() -> None:
    """Set up the logging format and level."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def create_save_forecast_batches(output_dir: str = None) -> None:
    setup_logging()
    current_dir = os.getcwd()
    data_dir = os.path.join(current_dir, 'results')
    config_path = os.path.join(current_dir, 'code', '.env')
    forecaster = TimeGPTForecaster(config_path = config_path, data_dir=data_dir, output_dir=output_dir)
    forecaster.forecast_all_batches()
