"""
run_preprocessing.py
====================
特徵工程入口腳本（對齊 sleep_classifiers/preprocessing_runner.py 功能）

此腳本讀取 sleep_classifiers/outputs/cropped/ 目錄中的原始裁切檔案
（*_cleaned_counts.out, *_cleaned_hr.out, *_cleaned_psg.out），
並對每位受試者執行 FeatureBuilder.build_subject_features()，
將結果輸出為 data/processed/{subject_id}.parquet。

執行方式：
    python run_preprocessing.py

選用引數：
    --cropped-dir   原始裁切檔案目錄（預設：sleep_classifiers/outputs/cropped）
    --output-dir    Parquet 輸出目錄（預設：data/processed）
    --subjects      指定要處理的受試者 ID，以空格分隔（預設：全部 31 位）
    --no-skip       強制重新計算，即使 parquet 已存在
"""

import argparse
import sys
import time
from pathlib import Path

# ── 確保 import 可找到 src ────────────────────────────────────────────────────
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.config import LEGACY_CROPPED_DIR, PROCESSED_DIR
from src.features.builder import FeatureBuilder


# ── 與 legacy SubjectBuilder.get_all_subject_ids() 完全對齊的受試者清單 ──────
ALL_SUBJECT_IDS: list[str] = [
    "3509524", "5132496", "1066528", "5498603", "2638030", "2598705",
    "5383425", "1455390", "4018081", "9961348", "1449548", "8258170",
    "781756",  "9106476", "8686948", "8530312", "3997827", "4314139",
    "1818471", "4426783", "8173033", "7749105", "5797046", "759667",
    "8000685", "6220552", "844359",  "9618981", "1360686", "46343",
    "8692923",
]


def run_preprocessing(
    subject_ids: list,
    cropped_dir: Path,
    output_dir: Path,
    skip_existing: bool = True,
) -> None:
    """
    對每位受試者執行特徵工程，並儲存 Parquet。

    Parameters
    ----------
    subject_ids   : 要處理的受試者 ID 清單
    cropped_dir   : 原始裁切 .out 檔案所在目錄
    output_dir    : Parquet 輸出目錄
    skip_existing : True 時跳過已有 parquet 的受試者
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    total = len(subject_ids)
    succeeded, skipped, failed = 0, 0, 0

    print(f"\n{'=' * 60}")
    print(f"  睡眠特徵工程前處理管線")
    print(f"  受試者總數   : {total}")
    print(f"  原始資料目錄 : {cropped_dir}")
    print(f"  輸出目錄     : {output_dir}")
    print(f"{'=' * 60}\n")

    t_global_start = time.perf_counter()

    for i, subject_id in enumerate(subject_ids, start=1):
        parquet_path = output_dir / f"{subject_id}.parquet"

        # ── 跳過已存在的受試者 ───────────────────────────────────────────
        if skip_existing and parquet_path.is_file():
            print(f"[{i:02d}/{total}] 受試者 {subject_id} — 已存在，跳過 ✓")
            skipped += 1
            continue

        # ── 建構原始檔案路徑 ─────────────────────────────────────────────
        counts_path = cropped_dir / f"{subject_id}_cleaned_counts.out"
        hr_path     = cropped_dir / f"{subject_id}_cleaned_hr.out"
        psg_path    = cropped_dir / f"{subject_id}_cleaned_psg.out"

        # 檢查必要檔案是否存在
        missing = [p for p in (counts_path, hr_path, psg_path) if not p.is_file()]
        if missing:
            print(f"[{i:02d}/{total}] 受試者 {subject_id} — 缺少原始檔案，跳過 ✗")
            for m in missing:
                print(f"             找不到：{m}")
            failed += 1
            continue

        # ── 執行特徵工程 ─────────────────────────────────────────────────
        print(f"[{i:02d}/{total}] 處理受試者 {subject_id}...", end="", flush=True)
        t_start = time.perf_counter()

        try:
            out = FeatureBuilder.build_subject_features(
                subject_id=subject_id,
                raw_counts_path=counts_path,
                raw_hr_path=hr_path,
                raw_psg_path=psg_path,
                output_dir=output_dir,
            )
            elapsed = time.perf_counter() - t_start
            print(f" 完成 ({elapsed:.1f}s) → {out.name}")
            succeeded += 1

        except Exception as exc:
            elapsed = time.perf_counter() - t_start
            print(f" 失敗 ({elapsed:.1f}s) ✗")
            print(f"             錯誤：{exc}")
            failed += 1

    # ── 最終摘要 ─────────────────────────────────────────────────────────────
    t_total = time.perf_counter() - t_global_start
    print(f"\n{'=' * 60}")
    print(f"  前處理完成：成功 {succeeded}，跳過 {skipped}，失敗 {failed}")
    print(f"  總耗時：{t_total / 60:.2f} 分鐘")
    print(f"{'=' * 60}\n")

    if failed > 0:
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="睡眠分類器特徵工程前處理管線（對齊 preprocessing_runner.py）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--cropped-dir",
        type=str,
        default=str(LEGACY_CROPPED_DIR),
        help="原始裁切 .out 檔案目錄",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROCESSED_DIR),
        help="Parquet 特徵輸出目錄",
    )
    parser.add_argument(
        "--subjects",
        type=str,
        nargs="+",
        default=None,
        metavar="SUBJECT_ID",
        help="指定受試者 ID（留空則處理全部 31 位）",
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        default=False,
        help="強制重新計算，即使 parquet 已存在",
    )

    args = parser.parse_args()

    subject_ids = args.subjects if args.subjects else ALL_SUBJECT_IDS

    # 驗證使用者指定的 ID 是否合法
    unknown = [sid for sid in subject_ids if sid not in ALL_SUBJECT_IDS]
    if unknown:
        print(f"[!] 警告：以下受試者 ID 不在已知清單中，仍會嘗試處理：{unknown}")

    run_preprocessing(
        subject_ids=subject_ids,
        cropped_dir=Path(args.cropped_dir),
        output_dir=Path(args.output_dir),
        skip_existing=not args.no_skip,
    )


if __name__ == "__main__":
    main()
