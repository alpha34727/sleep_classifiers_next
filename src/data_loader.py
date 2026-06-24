from pathlib import Path
from typing import List, Tuple
import numpy as np
import polars as pl

from src.config import FEATURE_COLS, TARGET_COL, BINARY_LABEL_MAP, THREE_CLASS_LABEL_MAP

class DataLoader:
    @staticmethod
    def load_subject_data(subject_id: str, processed_dir: Path) -> pl.DataFrame:
        """載入單一受試者的 Parquet 特徵數據"""
        parquet_path = processed_dir / f"{subject_id}.parquet"
        if not parquet_path.is_file():
            raise FileNotFoundError(f"Parquet file for subject {subject_id} not found at {parquet_path}")
        return pl.read_parquet(parquet_path)

    @staticmethod
    def load_all_subjects(processed_dir: Path,
                          subject_ids: List[str],
                          n_classes: int = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        載入並合併所有指定受試者的特徵數據，並將標籤映射為二分類或三分類。
        
        傳回：
            X (np.ndarray): 特徵矩陣，Shape: (N_samples, N_features)
            y (np.ndarray): 標籤陣列，Shape: (N_samples,)
            groups (np.ndarray): 樣本所屬的受試者 ID 陣列，Shape: (N_samples,)
        """
        if n_classes not in (2, 3):
            raise ValueError("n_classes must be either 2 or 3")

        label_map = BINARY_LABEL_MAP if n_classes == 2 else THREE_CLASS_LABEL_MAP

        all_dfs = []
        for subj in subject_ids:
            try:
                df = DataLoader.load_subject_data(subj, processed_dir)
                # 新增一列用以記錄受試者 ID
                df = df.with_columns(pl.lit(subj).alias("subject_id"))
                all_dfs.append(df)
            except FileNotFoundError:
                # 容錯：若某個受試者沒有離線特徵檔案，則跳過
                continue

        if not all_dfs:
            raise ValueError("No valid subject feature files were loaded.")

        # 合併所有受試者的 DataFrame
        df_all = pl.concat(all_dfs)

        # 映射標籤
        # 使用 replace 方法進行轉換
        df_all = df_all.with_columns(
            pl.col(TARGET_COL).replace(label_map).alias("mapped_stage")
        )

        # 過濾掉無法映射的 unscored 或無效標籤
        df_all = df_all.filter(pl.col("mapped_stage").is_in(list(set(label_map.values()))))

        # 提取特徵、標籤和群組
        X = df_all.select(FEATURE_COLS).to_numpy()
        y = df_all["mapped_stage"].to_numpy().astype(np.int64)
        groups = df_all["subject_id"].to_numpy()

        return X, y, groups
