import os
import glob
import pandas as pd
import numpy as np
import logging

def build_solution_forecast(forecast_df, output_npz, time_steps=92, expected_batch=67548):
    # Step 1: Sort and trim
    forecast_df_sorted = forecast_df.sort_values(["unique_id", "ds"])
    forecast_trimmed = forecast_df_sorted.groupby("unique_id").head(time_steps)

    # Step 2: Group and filter exact shape [batch, 92]
    forecast_grouped = (
        forecast_trimmed.groupby("unique_id")["TimeGPT"]
        .apply(lambda x: x.tolist())
        .loc[lambda x: x.apply(len) == time_steps]
    )

    # Step 3: Validate and sort to fixed size
    assert len(forecast_grouped) >= expected_batch, \
        f"Expected at least {expected_batch} valid series, got {len(forecast_grouped)}"
    
    forecast_grouped = forecast_grouped.sort_index().iloc[:expected_batch]
    forecast_array = np.stack(forecast_grouped.tolist()).astype(np.float32)

    #forecast_array = np.stack(forecast_grouped.tolist())  # shape: [batch, 92]

    # Step 4: Save
    np.savez_compressed(output_npz, kndvi=forecast_array)
    size_mb = os.path.getsize(output_npz) / (1024**2)
    logging.info(f"Saved to {output_npz} with shape {forecast_array.shape} (Size: {size_mb:.2f} MB)")

    # Step 5: Final check
    assert forecast_array.shape == (expected_batch, time_steps), "Shape mismatch."
    assert 20.0 <= size_mb <= 22.0, "File size outside expected ~21MB range."
    print("Validation passed: shape and size OK.")
