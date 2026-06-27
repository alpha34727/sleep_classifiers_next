import polars as pl
from pathlib import Path
from typing import List
from sleep_next.config import settings

SUBJECT_IDS = [
    "3509524", "5132496", "1066528", "5498603", "2638030", "2598705", "5383425",
    "1455390", "4018081", "9961348", "1449548", "8258170", "781756", "9106476",
    "8686948", "8530312", "3997827", "4314139", "1818471", "4426783", "8173033",
    "7749105", "5797046", "759667", "8000685", "6220552", "844359", "9618981",
    "1360686", "46343", "8692923"
]

def get_all_subject_ids() -> List[str]:
    return SUBJECT_IDS

def load_raw_psg(subject_id: str) -> pl.DataFrame:
    path = settings.DATA_DIR / "labels" / f"{subject_id}_labeled_sleep.txt"
    # Legacy PSG reader parses space-separated file with cols: timestamp, stage
    # It converts them to floats/ints.
    df = pl.read_csv(
        path,
        has_header=False,
        separator=" ",
        new_columns=["timestamp", "stage"],
        schema_overrides={"timestamp": pl.Float64, "stage": pl.Int32}
    )
    return df

def load_raw_motion(subject_id: str) -> pl.DataFrame:
    path = settings.DATA_DIR / "motion" / f"{subject_id}_acceleration.txt"
    df = pl.read_csv(
        path,
        has_header=False,
        separator=" ",
        new_columns=["timestamp", "x", "y", "z"],
        schema_overrides={"timestamp": pl.Float64, "x": pl.Float32, "y": pl.Float32, "z": pl.Float32}
    )
    # Remove repeats (same timestamp)
    df = df.unique(subset=["timestamp"], keep="first", maintain_order=True)
    return df

def load_raw_heart_rate(subject_id: str) -> pl.DataFrame:
    path = settings.DATA_DIR / "heart_rate" / f"{subject_id}_heartrate.txt"
    df = pl.read_csv(
        path,
        has_header=False,
        separator=",",
        new_columns=["timestamp", "heart_rate"],
        schema_overrides={"timestamp": pl.Float64, "heart_rate": pl.Float32}
    )
    df = df.unique(subset=["timestamp"], keep="first", maintain_order=True)
    return df
