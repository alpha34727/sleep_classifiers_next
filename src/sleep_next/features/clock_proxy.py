import numpy as np
import polars as pl
from pathlib import Path
from sleep_next.config import settings

def cosine_proxy(time_seconds: float) -> float:
    sleep_drive_cosine_shift = 5.0
    seconds_per_hour = 3600.0
    seconds_per_day = 86400.0
    val = -1.0 * np.cos(
        (time_seconds - sleep_drive_cosine_shift * seconds_per_hour)
        * 2.0 * np.pi / seconds_per_day
    )
    return float(val)

def build_time(valid_epoch_timestamps: np.ndarray) -> np.ndarray:
    if len(valid_epoch_timestamps) == 0:
        return np.empty((0,), dtype=np.float32)
    first_timestamp = valid_epoch_timestamps[0]
    # Units to hours
    relative_hours = (valid_epoch_timestamps - first_timestamp) / 3600.0
    return relative_hours.astype(np.float32)

def build_cosine(valid_epoch_timestamps: np.ndarray) -> np.ndarray:
    if len(valid_epoch_timestamps) == 0:
        return np.empty((0,), dtype=np.float32)
    first_timestamp = valid_epoch_timestamps[0]
    cosine_features = []
    for ts in valid_epoch_timestamps:
        val = cosine_proxy(ts - first_timestamp)
        cosine_features.append(val)
    return np.array(cosine_features, dtype=np.float32)

def build_circadian_model(subject_id: str, valid_epoch_timestamps: np.ndarray) -> np.ndarray:
    if len(valid_epoch_timestamps) == 0:
        return np.empty((0, 1), dtype=np.float32)
        
    circadian_file = settings.DATA_DIR / "circadian_predictions" / f"{subject_id}_clock_proxy.txt"
    if not circadian_file.is_file():
        # Fallback or empty if not included
        return np.zeros((len(valid_epoch_timestamps), 1), dtype=np.float32)
        
    # Read CSV
    circ_df = pl.read_csv(circadian_file, has_header=False, separator=",")
    circ_array = circ_df.to_numpy() # cols: timestamp, prediction
    
    first_value = float(np.interp(valid_epoch_timestamps[0], circ_array[:, 0], circ_array[:, 1]))
    
    features = []
    min_diff = float(np.amin(circ_array[:, 1] - first_value))
    if abs(min_diff) < 1e-8:
        min_diff = -1.0 # prevent division by zero
        
    for ts in valid_epoch_timestamps:
        val = float(np.interp(ts, circ_array[:, 0], circ_array[:, 1]))
        normalized_value = (val - first_value) / min_diff
        if normalized_value < settings.LOWER_BOUND:
            normalized_value = settings.LOWER_BOUND
        features.append([normalized_value])
        
    return np.array(features, dtype=np.float32)
