# Production-Ready Generative Pretrained Transformer Model Submission

This repository provides a solution for the **kNDVI Prediction Challenge**, focusing on forecasting vegetation dynamics using machine learning and deep time series models with TimeGPT. It includes data extraction, preprocessing, batch forecasting, evaluation, and solution formatting designed for local, or HPC execution.

## Table of Contents

* [Project Structure](#project-structure)
* [Setup and Installation](#setup-and-installation)
* [Execution](#execution)
* [Modules Overview](#modules-overview)

  * [1. `load_export_data.py`](#1-load_export_datapy)
  * [2. `preprocess.py`](#2-preprocesspy)
  * [3. `forecast.py`](#3-forecastpy)
  * [4. `forecast_evaluator.py`](#4-forecast_evaluatorpy)
  * [5. `write_solution.py`](#5-write_solutionpy)
  * [6. `orchestrator.py`](#6-orchestratorpy)
* [Output Files](#output-files)
* [Temporary Directory Handling](#temporary-directory-handling)
* [Notes](#notes)

---

## Project Structure

```
.
├── code/
│   ├── load_export_data.py
│   ├── preprocess.py
│   ├── forecast.py
│   ├── forecast_evaluator.py
│   ├── write_solution.py
│   └── orchestrator.py
├── results/                # Output directory for npz/parquet/solution files
├── notebooks/              # Exploratory data analysis inside, baseline ridge
├── docs/                   # Report
├── logs/                   # Logs stored
├── environment.yml         # Conda environment definition
├── requirements.txt        # (optional) pip-based installation
├── info.toml        
├── run.sh                  # HPC execution file
├── README.md
```

## Setup and Installation

Python Version, **Python 3.11.7**

### Option 1: Using Conda (recommended)

Create and activate the environment from the provided `environment.yml`:

```bash
conda env create -f environment.yml
conda activate time_gpt_env
```

This ensures compatibility with all required versions and packages.

### Option 2: Using pip

```bash
python -m venv kndvi_venv
source kndvi_venv/bin/activate  # On Windows: .\kndvi_venv\Scripts\activate
pip install -r requirements.txt
```

## Execution

You can run the full pipeline by executing:

```bash
python code/orchestrator.py
```

Or by executing:
```bash
run.sh
```

Or specify an output directory:

```bash
python code/orchestrator.py --output_dir /home/user/results
```

This executes the following pipeline:

1. Load `.npz` datasets and extract the `kndvi` variable.
2. Preprocess and export data to `.parquet` format.
3. Generate batch forecasts using TimeGPT.
4. Evaluate training and validation performance.
5. Format and export results to `.toml`.

---

## Modules Overview

### 1. `load_export_data.py`

* Loads `.npz` files and extracts the `kndvi` variable.
* Defines the `KNDVIExtractor` class and `get_default_paths()` function.
* Supports structured output to avoid hardcoded paths.

### 2. `preprocess.py`

* Converts extracted NumPy arrays to the required `float32` format.
* Ensures input is ready for forecasting models.

### 3. `forecast.py`

* Implements forecasting using the Nixtla TimeGPT API.
* Accepts model parameters like `horizon`, `context_window`, `finetune_depth`.
* Forecasting is done in batches using `unique_id` for time series grouping.

### 4. `forecast_evaluator.py`

* Calculates MAE and RMSE using predicted and actual values.
* Stores performance metrics in memory and/or in a `.toml` report.

### 5. `write_solution.py`

* Generates the final challenge submission file.
* Updates or creates a solution metadata file in TOML format.

### 6. `orchestrator.py`

* Main control script for end-to-end execution.
* Supports temporary directories (`tempfile`) if no `--output_dir` is given.
* Automatically performs cleanup if temp directory is used.

---

## Output Files

Typical files created in the output directory:

* `train_kndvi.npz`, `val_kndvi.npz` – raw extracted input data.
* `train_timegpt.parquet`, `val_timegpt.parquet` `trainerrors_timegpt.parquet` – forecasts for training/validation.
* `info.toml` – metadata and metrics for submission.
* `solution.npz` – formatted predictions for the challenge.

---

## Temporary Directory Handling

If no `--output_dir` is specified, a temporary directory is created using Python’s `tempfile.TemporaryDirectory()` context manager. It is cleaned up automatically at the end of the pipeline.

To persist results, pass an explicit path:

```bash
python code/orchestrator.py --output_dir ./results
```

---

## Notes

* Forecasting with TimeGPT requires a valid **API key** stored in the environment as `TIMEGPT_API_KEY`.
* The project supports modular extension for other models (e.g., ARIMA, LSTM).
* All batch operations are memory-aware and optimized for HPC clusters.
* You can call individual module functions for debugging or testing each stage separately.
