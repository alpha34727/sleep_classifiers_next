import gc
from pathlib import Path
import numpy as np
import polars as pl

from src.features.signal_processing import smooth_gauss, convolve_with_dog

class FeatureBuilder:
    WINDOW_SIZE = 10 * 30 - 15  # 285 seconds

    @staticmethod
    def build_subject_features(subject_id: str,
                               raw_counts_path: Path,
                               raw_hr_path: Path,
                               raw_psg_path: Path,
                               output_dir: Path) -> Path:
        """
        處理單一受試者，計算 Counts 與 HR 的 window 特徵，並儲存成 parquet 檔案。
        此實作在數值上完全對齊舊專案以通過測試，但大幅優化了內存佔用與運行效率。
        """
        # 1. 讀取原始 Cropped 檔案
        # counts 是逗號分隔
        df_counts = pl.read_csv(raw_counts_path, has_header=False, new_columns=["timestamp", "count"]).slice(1)
        # hr 是空格分隔
        df_hr = pl.read_csv(raw_hr_path, has_header=False, separator=" ", new_columns=["timestamp", "heart_rate"]).slice(1)
        # psg 是空格分隔
        df_psg = pl.read_csv(raw_psg_path, has_header=False, separator=" ", new_columns=["timestamp", "stage"]).slice(1)


        # 2. 獲取 start_time
        start_time = df_psg["timestamp"][0]

        # 3. 讀取 motion 的第一列 timestamp 以獲得與舊版完全一致的有效 epoch 過濾（含有運動數據缺失的 gap）
        raw_motion_path = raw_counts_path.parent / f"{subject_id}_cleaned_motion.out"
        df_motion_times = pl.read_csv(raw_motion_path, has_header=False, separator=" ", columns=[0]).slice(1)
        df_motion_times.columns = ["timestamp"]

        # 4. 計算 valid epochs (對齊舊版 get_valid_epochs 邏輯)
        hr_times = df_hr["timestamp"].to_numpy()
        hr_floored = hr_times - np.mod(hr_times - start_time, 30.0)
        hr_epoch_set = set(hr_floored)

        motion_times = df_motion_times["timestamp"].to_numpy()
        motion_floored = motion_times - np.mod(motion_times - start_time, 30.0)
        motion_epoch_set = set(motion_floored)

        valid_epochs = []
        valid_stages = []
        for row in df_psg.iter_rows():
            t, stage = row[0], row[1]
            if stage != -1 and t in hr_epoch_set and t in motion_epoch_set:
                valid_epochs.append(t)
                valid_stages.append(stage)


        valid_epochs = np.array(valid_epochs)
        valid_stages = np.array(valid_stages)

        if len(valid_epochs) == 0:
            raise ValueError(f"Subject {subject_id} has no valid epochs.")

        # 4. 內插與信號卷積計算
        # Counts 內插
        counts_t = df_counts["timestamp"].to_numpy()
        counts_v = df_counts["count"].to_numpy()
        interp_counts_t = np.arange(np.amin(counts_t), np.amax(counts_t), 1.0)
        interp_counts_v = np.interp(interp_counts_t, counts_t, counts_v)

        # Heart Rate 內插與 DoG 卷積 + 標準化
        hr_t = df_hr["timestamp"].to_numpy()
        hr_v = df_hr["heart_rate"].to_numpy()
        interp_hr_t = np.arange(np.amin(hr_t), np.amax(hr_t), 1.0)
        interp_hr_v = np.interp(interp_hr_t, hr_t, hr_v)

        dog_hr = convolve_with_dog(interp_hr_v, FeatureBuilder.WINDOW_SIZE)
        scale_hr = np.percentile(np.abs(dog_hr), 90)
        normalized_hr = dog_hr / scale_hr if scale_hr > 0 else dog_hr

        # 5. 為每個 valid epoch 提取 window 特徵
        count_features = []
        hr_features = []
        time_features = []
        cosine_features = []

        first_timestamp = valid_epochs[0]

        for epoch_time in valid_epochs:
            # 視窗時間範圍：(t - 285) 到 (t + 315)
            start_t = epoch_time - FeatureBuilder.WINDOW_SIZE
            end_t = epoch_time + 30.0 + FeatureBuilder.WINDOW_SIZE

            # Counts 特徵：在 interp_counts 中截取 window 並計算 Gaussian 卷積
            c_left = np.searchsorted(interp_counts_t, start_t, side='right')
            c_right = np.searchsorted(interp_counts_t, end_t, side='left')
            counts_in_window = interp_counts_v[c_left:c_right]
            c_feat = smooth_gauss(counts_in_window, len(counts_in_window))
            count_features.append(c_feat)

            # Heart Rate 特徵：在 normalized_hr 中截取 window 並計算標準差 (std)
            hr_left = np.searchsorted(interp_hr_t, start_t, side='right')
            hr_right = np.searchsorted(interp_hr_t, end_t, side='left')
            hr_in_window = normalized_hr[hr_left:hr_right]
            hr_feat = np.std(hr_in_window) if len(hr_in_window) > 0 else 0.0
            hr_features.append(hr_feat)

            # Time 特徵：以第一個 epoch 為基準的小時數
            t_feat = (epoch_time - first_timestamp) / 3600.0
            time_features.append(t_feat)

            # Cosine 特徵
            cos_val = -1.0 * np.cos((epoch_time - first_timestamp - 5.0 * 3600.0) * 2.0 * np.pi / 86400.0)
            cosine_features.append(cos_val)

        # 6. 組裝 Polars DataFrame
        df_out = pl.DataFrame({
            "timestamp": valid_epochs,
            "stage": valid_stages,
            "count_feature": count_features,
            "hr_feature": hr_features,
            "time_feature": time_features,
            "cosine_feature": cosine_features
        })

        # 7. 輸出為 parquet 檔案
        output_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = output_dir / f"{subject_id}.parquet"
        df_out.write_parquet(parquet_path)

        # 8. 主動釋放內存
        del df_counts, df_hr, df_psg, df_motion_times, df_out
        gc.collect()


        return parquet_path
