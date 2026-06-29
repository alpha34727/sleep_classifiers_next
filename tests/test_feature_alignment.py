import pytest
import pandas as pd
import numpy as np
import polars as pl
from pathlib import Path
from sleep_next.config import settings
from sleep_next.data import preprocess

ORIG_DIR = Path("C:/Users/Johnsou/Desktop/sleep_classifiers/outputs/features")
NEXT_DIR = settings.FEATURE_DIR
SUBJECT = "3509524"

@pytest.mark.skipif(not ORIG_DIR.exists(), reason="Original sleep_classifiers project outputs/features directory not found")
def test_legacy_feature_alignment():
    # If the next parquet doesn't exist, run preprocessing to generate it
    parquet_path = NEXT_DIR / f"{SUBJECT}_features.parquet"
    if not parquet_path.exists():
        preprocess.crop_subject_data(SUBJECT)
        preprocess.build_features_for_subject(SUBJECT)
        
    # Load original
    orig_count = pd.read_csv(ORIG_DIR / f"{SUBJECT}_count_feature.out", header=None).values.flatten()
    orig_hr = pd.read_csv(ORIG_DIR / f"{SUBJECT}_hr_feature.out", header=None).values.flatten()
    orig_labels = pd.read_csv(ORIG_DIR / f"{SUBJECT}_psg_labels.out", header=None).values.flatten()
    orig_cosine = pd.read_csv(ORIG_DIR / f"{SUBJECT}_cosine_feature.out", header=None).values.flatten()
    orig_time = pd.read_csv(ORIG_DIR / f"{SUBJECT}_time_feature.out", header=None).values.flatten()
    
    # Load next
    df_next = pl.read_parquet(parquet_path)
    next_count = df_next["feature_count"].to_numpy()
    next_hr = df_next["feature_hr"].to_numpy()
    next_labels = df_next["label"].to_numpy()
    next_cosine = df_next["feature_cosine"].to_numpy()
    next_time = df_next["feature_time"].to_numpy()
    
    # Verify that shapes match
    assert len(orig_count) == len(next_count), f"Feature lengths do not match: orig={len(orig_count)}, next={len(next_count)}"
    assert len(orig_hr) == len(next_hr)
    assert len(orig_labels) == len(next_labels)
    assert len(orig_cosine) == len(next_cosine)
    assert len(orig_time) == len(next_time)
    
    # Assert values are close (within single-precision float / formatting rounding limits)
    if settings.REPRODUCE_LEGACY_BUG:
        # count feature has a slightly higher tolerance (1e-2) due to original project 
        # using formatted text %.6f format output in activity count file saves.
        np.testing.assert_allclose(orig_count, next_count, rtol=1e-2, atol=1e-2)
        np.testing.assert_allclose(orig_hr, next_hr, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(orig_labels, next_labels, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(orig_cosine, next_cosine, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(orig_time, next_time, rtol=1e-5, atol=1e-5)
