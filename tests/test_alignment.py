import sys
import tempfile
from pathlib import Path

# 將專案根目錄加入 sys.path 以防 ModuleNotFoundError
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import numpy as np
import polars as pl
import pytest

from src.features.builder import FeatureBuilder


def test_subject_feature_alignment():
    """
    驗證新 Polars/NumPy pipeline 計算出的特徵值，是否與舊專案輸出檔案中的特徵值完美對齊。
    """
    subject_id = "1066528"
    project_root = Path(__file__).parent.parent
    
    # 原始 cropped 資料路徑
    raw_counts_path = project_root / f"sleep_classifiers/outputs/cropped/{subject_id}_cleaned_counts.out"
    raw_hr_path = project_root / f"sleep_classifiers/outputs/cropped/{subject_id}_cleaned_hr.out"
    raw_psg_path = project_root / f"sleep_classifiers/outputs/cropped/{subject_id}_cleaned_psg.out"
    
    # 舊專案預先算好的特徵路徑
    legacy_counts_path = project_root / f"sleep_classifiers/outputs/features/{subject_id}_count_feature.out"
    legacy_hr_path = project_root / f"sleep_classifiers/outputs/features/{subject_id}_hr_feature.out"
    legacy_time_path = project_root / f"sleep_classifiers/outputs/features/{subject_id}_time_feature.out"
    legacy_cosine_path = project_root / f"sleep_classifiers/outputs/features/{subject_id}_cosine_feature.out"
    
    # 載入舊專案預先算好的特徵陣列
    legacy_counts = np.loadtxt(legacy_counts_path)
    legacy_hr = np.loadtxt(legacy_hr_path)
    legacy_time = np.loadtxt(legacy_time_path)
    legacy_cosine = np.loadtxt(legacy_cosine_path)
    
    # 使用臨時目錄來儲存 Parquet 檔案
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        
        # 執行新的 FeatureBuilder
        parquet_path = FeatureBuilder.build_subject_features(
            subject_id=subject_id,
            raw_counts_path=raw_counts_path,
            raw_hr_path=raw_hr_path,
            raw_psg_path=raw_psg_path,
            output_dir=output_dir
        )
        
        # 讀取輸出的 Parquet 檔案
        df_new = pl.read_parquet(parquet_path)
        
        # 取得新計算的各特徵欄位
        new_counts = df_new["count_feature"].to_numpy()
        new_hr = df_new["hr_feature"].to_numpy()
        new_time = df_new["time_feature"].to_numpy()
        new_cosine = df_new["cosine_feature"].to_numpy()
        
        # 1. 驗證長度是否相同
        assert len(new_counts) == len(legacy_counts), f"Length mismatch: {len(new_counts)} vs {len(legacy_counts)}"
        
        # 2. 驗證各項特徵的數值對齊 (容許極小浮點數誤差)
        np.testing.assert_allclose(
            new_counts, legacy_counts, rtol=1e-5, atol=1e-5,
            err_msg="Counts features do not align!"
        )
        
        np.testing.assert_allclose(
            new_hr, legacy_hr, rtol=1e-5, atol=1e-5,
            err_msg="Heart rate features do not align!"
        )
        
        np.testing.assert_allclose(
            new_time, legacy_time, rtol=1e-5, atol=1e-5,
            err_msg="Time features do not align!"
        )
        
        np.testing.assert_allclose(
            new_cosine, legacy_cosine, rtol=1e-5, atol=1e-5,
            err_msg="Cosine features do not align!"
        )
        
        print("All features align perfectly!")
