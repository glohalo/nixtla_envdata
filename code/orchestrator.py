from load_export_data import KNDVIExtractor, get_default_paths
from preprocess import TimeGPTDataProcessor
from forecast import create_save_forecast_batches
from forecast_evaluator import ForecastEvaluator
from write_solution import build_solution_forecast
import os
import logging
import tempfile
logging.basicConfig(level=logging.INFO,format="%(asctime)s - %(levelname)s - %(message)s")
#from preprocess import convert_store_data

def main() -> None:
    """
    Function that integrates and orchestrate the process to create predictions
    Includes the next modules:
    - load_export_data, get_default_paths
    - preprocess, TimeGPTDataProcessor
    - forecast, create_save_forecast_batches
    - forecast_evaluator, ForecastEvaluator
    - write_solution, build_solution_forecast
    """
    current_dir = os.getcwd()
    output_path = os.path.join(current_dir, 'results')
    documentation_dir = os.path.join(current_dir, 'docs')
    documentation_path = os.path.join(documentation_dir, 'report.pdf')
    output_npz = os.path.join(output_path, "solution.npz")
    output_backup = os.path.join(output_path, "solution_backup.npz")
    logging.info(f'Output path: {output_path}')

    train, val, test, results = get_default_paths()
    extractor = KNDVIExtractor(train, val, test, results)
    extractor.extract_store()
    logging.info("Data extracted and saved!")
    #### processing data ######    
    processor = TimeGPTDataProcessor(train, val, test, results, output_path)
    processor.load_data()
    processor.save_to_parquet()
    logging.info("Data processed!")

    with tempfile.TemporaryDirectory() as forecast_tmp_dir:
        logging.info(f"Temporal dir for forecast, {forecast_tmp_dir}")
        create_save_forecast_batches(output_dir=forecast_tmp_dir)
        logging.info("Forecast saved!")
        ### Evaluate forecast ###### 
        evaluator = ForecastEvaluator(
            forecast_tmp_dir,
            val_path=f"{results}/val_timegpt.parquet",
            train_path=f"{results}/trainerrors_timegpt.parquet"
        )
        evaluator.load_data()
        try:
            evaluator.evaluate_validation()
            evaluator.evaluate_training()
        finally:
            evaluator.cleanup()
        # Store the solution forecast
        forecast_df = evaluator.forecast_df.sort_values(["unique_id", "ds"])
        print(forecast_df.head())
        build_solution_forecast(forecast_df, output_npz)
        # Write info on toml file
        evaluator.update_errors_in_toml(path=f"{current_dir}/info.toml", solution_path=output_npz, documentation_path= documentation_path, backup_path= output_backup)
        try:
            os.remove(f"{results}/val_timegpt.parquet")
            os.remove(f"{results}/train_timegpt.parquet")
            os.remove(f"{results}/train_kndvi.npz")
            os.remove(f"{results}/val_kndvi.npz")
            os.remove(f"{results}/test_nosolution.npz")
            logging.info("Temporary evaluation files deleted.")
        except FileNotFoundError:
            logging.warning("Some evaluation files were already deleted or not found.")

if __name__ == "__main__":
    main()

