import os
import time
import psutil
import logging
import numpy as np
from typing import Tuple, Final

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
class KNDVIExtractor:
    def __init__(self,
                 train_path: str,
                 val_path: str,
                 test_path: str,
                 results_dir: str,
                 index_variable: int = 4):
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.results_dir = results_dir
        self.index_variable = index_variable

    @staticmethod
    def monitor(func):
        def wrapper(*args, **kwargs):
            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss / (1024 ** 2)
            start = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            mem_after = process.memory_info().rss / (1024 ** 2)
            cpu_percent = process.cpu_percent(interval=0.1)
            logging.info("\nMONITORING REPORT")
            logging.info(f"Time elapsed:     {elapsed:.2f} seconds")
            logging.info(f"Memory used:     {mem_after - mem_before:.2f} MB")
            logging.info(f"CPU usage:       {cpu_percent:.2f}%")
            return result
        return wrapper

    @monitor.__func__
    def extract_store(self, train: bool = True, val: bool = True, test: bool = True) -> None:
        """Extract the kNDVI variable and save it from .npz datasets."""
        logging.info("Starting to extract and store the data!")

        if train:
            with np.load(self.train_path) as data:
                kndvi_train = data['train'][:, :, self.index_variable]
                np.savez_compressed(os.path.join(self.results_dir, "train_kndvi.npz"), kndvi=kndvi_train)
            logging.info("Train data stored!")

        if val:
            with np.load(self.val_path) as data:
                kndvi_val = data['validation'][:, :, self.index_variable]
                np.savez_compressed(os.path.join(self.results_dir, "val_kndvi.npz"), kndvi=kndvi_val)
            logging.info("Validation data stored!")

        if test:
            with np.load(self.test_path) as data:
                kndvi_test = data['test'][:, :, self.index_variable]
                np.savez_compressed(os.path.join(self.results_dir, "test_nosolution.npz"), kndvi=kndvi_test)
            logging.info("Test data stored!")

    def load_stored_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load stored kNDVI training and validation datasets."""
        path_train = os.path.join(self.results_dir, "train_kndvi.npz")
        path_val = os.path.join(self.results_dir, "val_kndvi.npz")
        logging.info(f"Loading train data from {path_train}")
        logging.info(f"Loading validation data from {path_val}")
        train_kndvi = np.load(path_train)
        val_kndvi = np.load(path_val)
        #train_kndvi = np.load(path_train)['kndvi']
        #val_kndvi = np.load(path_val)['kndvi']
        return train_kndvi, val_kndvi


def get_default_paths() -> Tuple[str, str, str, str]:
    """Returns default dataset and result paths."""
    current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    project_root = os.path.dirname(current_dir)
    results_path = os.path.join(project_root, "results")
    return (
        "/work/gk62kagy-kndvi_prediction_challenge_data/train.npz",
        "/work/gk62kagy-kndvi_prediction_challenge_data/validation.npz",
        "/work/gk62kagy-kndvi_prediction_challenge_data/test_nosolution.npz",
        results_path
    )
