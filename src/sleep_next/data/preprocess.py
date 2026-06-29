import numpy as np
import polars as pl
from pathlib import Path
from sleep_next.config import settings
from sleep_next.data import loader
from sleep_next.features import motion, clock_proxy

def convolve_with_dog(y: np.ndarray, box_pts: int) -> np.ndarray:
    y = y - np.mean(y)
    box = np.ones(box_pts, dtype=np.float32)
    mu1 = int(box_pts / 2.0)
    sigma1 = 120.0
    mu2 = int(box_pts / 2.0)
    sigma2 = 600.0
    scalar = 0.75
    
    for ind in range(box_pts):
        box[ind] = np.exp(-0.5 * (((ind - mu1) / sigma1) ** 2)) - scalar * np.exp(
            -0.5 * (((ind - mu2) / sigma2) ** 2)
        )
        
    # Legacy insertion behavior
    y = np.insert(y, 0, np.flip(y[0:int(box_pts / 2)]))
    y = np.insert(y, len(y) - 1, np.flip(y[int(-box_pts / 2):]))
    y_smooth = np.convolve(y, box, mode='valid')
    return y_smooth

def smooth_gauss(y: np.ndarray, box_pts: int) -> float:
    box = np.ones(box_pts, dtype=np.float32)
    mu = int(box_pts / 2.0)
    sigma = 50.0  # seconds
    
    for ind in range(box_pts):
        box[ind] = np.exp(-0.5 * (((ind - mu) / sigma) ** 2))
        
    box = box / np.sum(box)
    return float(np.sum(box * y))

def crop_subject_data(subject_id: str):
    # Load raw data
    psg_df = loader.load_raw_psg(subject_id)
    motion_df = loader.load_raw_motion(subject_id)
    hr_df = loader.load_raw_heart_rate(subject_id)
    
    # Intersecting interval
    start_time = max(psg_df["timestamp"].min(), motion_df["timestamp"].min(), hr_df["timestamp"].min())
    end_time = min(psg_df["timestamp"].max(), motion_df["timestamp"].max(), hr_df["timestamp"].max())
    
    # Crop
    psg_cropped = psg_df.filter((pl.col("timestamp") >= start_time) & (pl.col("timestamp") < end_time))
    motion_cropped = motion_df.filter((pl.col("timestamp") >= start_time) & (pl.col("timestamp") < end_time))
    hr_cropped = hr_df.filter((pl.col("timestamp") >= start_time) & (pl.col("timestamp") < end_time))
    
    # Compute Python-based activity counts
    motion_np = motion_cropped.to_numpy() # timestamp, x, y, z
    counts_np = motion.compute_activity_counts(motion_np[:, 0], motion_np[:, 3])
    counts_df = pl.DataFrame(counts_np, schema=["timestamp", "count"])
    
    # Write cropped outputs as Parquet
    psg_cropped.write_parquet(settings.CROPPED_DIR / f"{subject_id}_cleaned_psg.parquet")
    motion_cropped.write_parquet(settings.CROPPED_DIR / f"{subject_id}_cleaned_motion.parquet")
    hr_cropped.write_parquet(settings.CROPPED_DIR / f"{subject_id}_cleaned_hr.parquet")
    counts_df.write_parquet(settings.CROPPED_DIR / f"{subject_id}_cleaned_counts.parquet")

def build_features_for_subject(subject_id: str):
    # Load cropped files
    psg_df = pl.read_parquet(settings.CROPPED_DIR / f"{subject_id}_cleaned_psg.parquet")
    hr_df = pl.read_parquet(settings.CROPPED_DIR / f"{subject_id}_cleaned_hr.parquet")
    counts_df = pl.read_parquet(settings.CROPPED_DIR / f"{subject_id}_cleaned_counts.parquet")
    
    if settings.REPRODUCE_LEGACY_BUG:
        psg_df = psg_df[1:]
        hr_df = hr_df[1:]
        counts_df = counts_df[1:]
        
    start_time = psg_df["timestamp"][0]
    
    # Compute valid epochs:
    # 30-second floored epochs from start_time
    # epoch timestamp is in both motion (counts) and heart rate intervals
    # and stage != -1 (unscored)
    motion_timestamps = counts_df["timestamp"].to_numpy()
    hr_timestamps = hr_df["timestamp"].to_numpy()
    
    # Get dictionaries of valid floored timestamps (as sets for O(1) checks)
    motion_epochs = set(((motion_timestamps - start_time) // 30) * 30 + start_time)
    hr_epochs = set(((hr_timestamps - start_time) // 30) * 30 + start_time)
    
    valid_psg = psg_df.filter(
        (pl.col("stage") != -1) &
        (pl.col("timestamp").is_in(list(motion_epochs))) &
        (pl.col("timestamp").is_in(list(hr_epochs)))
    )
    
    valid_epoch_timestamps = valid_psg["timestamp"].to_numpy()
    valid_labels = valid_psg["stage"].to_numpy()
    
    if len(valid_epoch_timestamps) == 0:
        return
        
    # --- 1. Compute Activity Count Feature ---
    # Interpolate counts at 1-second interval
    t_min = np.amin(motion_timestamps)
    t_max = np.amax(motion_timestamps)
    interpolated_t = np.arange(t_min, t_max, 1.0)
    interpolated_counts = np.interp(interpolated_t, motion_timestamps, counts_df["count"].to_numpy())
    
    count_window = 10 * 30 - 15 # 285
    count_features = []
    for ts in valid_epoch_timestamps:
        # Find indices within window: [ts - 285, ts + 30 + 285]
        start_w = ts - count_window
        end_w = ts + 30.0 + count_window
        idx = np.where((interpolated_t > start_w) & (interpolated_t < end_w))[0]
        vals = interpolated_counts[idx]
        count_features.append(smooth_gauss(vals, len(vals)))
        
    # --- 2. Compute Heart Rate Feature ---
    # Interpolate HR at 1-second interval
    hr_raw_t = hr_df["timestamp"].to_numpy()
    hr_raw_v = hr_df["heart_rate"].to_numpy()
    interpolated_hr_t = np.arange(np.amin(hr_raw_t), np.amax(hr_raw_t), 1.0)
    interpolated_hr_v = np.interp(interpolated_hr_t, hr_raw_t, hr_raw_v)
    
    # DOG filter convolve
    hr_window = 10 * 30 - 15 # 285
    smoothed_hr_v = convolve_with_dog(interpolated_hr_v, hr_window)
    # Scale by 90th percentile of abs values
    scale = np.percentile(np.abs(smoothed_hr_v), 90)
    if scale < 1e-8:
        scale = 1.0
    smoothed_hr_v = smoothed_hr_v / scale
    
    hr_features = []
    for ts in valid_epoch_timestamps:
        start_w = ts - hr_window
        end_w = ts + 30.0 + hr_window
        idx = np.where((interpolated_hr_t > start_w) & (interpolated_hr_t < end_w))[0]
        vals = smoothed_hr_v[idx]
        hr_features.append(np.std(vals))
        
    # --- 3. Time Based Features ---
    time_features = clock_proxy.build_time(valid_epoch_timestamps)
    cosine_features = clock_proxy.build_cosine(valid_epoch_timestamps)
    circadian_features = clock_proxy.build_circadian_model(subject_id, valid_epoch_timestamps)
    
    # Write features to Parquet
    features_df = pl.DataFrame({
        "timestamp": valid_epoch_timestamps.astype(np.float64),
        "label": valid_labels.astype(np.int32),
        "feature_count": np.array(count_features, dtype=np.float32),
        "feature_hr": np.array(hr_features, dtype=np.float32),
        "feature_time": time_features,
        "feature_cosine": cosine_features,
        "feature_circadian": circadian_features.flatten()
    })
    
    features_df.write_parquet(settings.FEATURE_DIR / f"{subject_id}_features.parquet")
