import argparse
import sys
from pathlib import Path
from typing import Dict, Type

import numpy as np

# 確保 import 可找到 src
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.config import PROCESSED_DIR, PROJECT_ROOT
from src.data_loader import DataLoader
from src.evaluation.validator import SleepValidator
from src.models.base import BaseSleepClassifier
from src.models.baseline import (
    LogisticRegressionClassifier,
    MLPClassifierWrapper,
    RandomForestClassifierWrapper,
    KNNClassifierWrapper,
)
from src.models.modern_gbdt import LightGBMClassifier, XGBoostClassifier


def run_evaluation(
    processed_dir: Path,
    n_splits: int,
    train_size: float,
    n_classes: int,
    n_jobs: int,
) -> None:
    # 1. 取得受試者列表
    subject_ids = sorted([p.stem for p in processed_dir.glob("*.parquet")])
    if not subject_ids:
        print(f"[!] 錯誤：在 {processed_dir} 下找不到任何 *.parquet 檔案。")
        print("    請先執行：python run_preprocessing.py")
        sys.exit(1)

    print(f"[*] 正在從 {processed_dir} 載入 {len(subject_ids)} 位受試者的特徵數據...")
    X, y, groups = DataLoader.load_all_subjects(
        processed_dir=processed_dir,
        subject_ids=subject_ids,
        n_classes=n_classes,
    )
    print(f"[*] 數據載入完成。樣本數：{X.shape[0]}，特徵數：{X.shape[1]}")

    # 2. 定義要評估的模型
    model_classes: Dict[str, Type[BaseSleepClassifier]] = {
        "Logistic Regression": LogisticRegressionClassifier,
        "Random Forest": RandomForestClassifierWrapper,
        "Neural Net (MLP)": MLPClassifierWrapper,
        "k-NN": KNNClassifierWrapper,
        "LightGBM": LightGBMClassifier,
        "XGBoost": XGBoostClassifier,
    }

    # 3. 執行評估
    results = {}
    print(f"\n[*] 開始進行 Monte Carlo 交叉驗證 ({n_splits} 折，並行度 n_jobs={n_jobs})...")
    
    for name, model_cls in model_classes.items():
        print(f"    -> 評估模型: {name:20s}", end="", flush=True)
        try:
            mccv_res = SleepValidator.run_mccv(
                model_cls=model_cls,
                X=X,
                y=y,
                groups=groups,
                n_splits=n_splits,
                train_size=train_size,
                n_classes=n_classes,
                n_jobs=n_jobs,
            )
            results[name] = mccv_res["mccv_summary"]
            print(" 完成 ✓")
        except Exception as e:
            print(f" 失敗 ✗ (錯誤: {e})")

    # 4. 列印 Markdown 格式的結果表格
    print("\n" + "=" * 80)
    print(f"   睡眠階段預測評估結果 (MCCV - {n_splits} Splits, {n_classes} 分類)")
    print("=" * 80)
    
    if n_classes == 2:
        print(f"| 模型名稱 | Accuracy (Mean ± Std) | Cohen's Kappa (Mean ± Std) | Sensitivity (Mean ± Std) | Specificity (Mean ± Std) | AUC-ROC (Mean ± Std) |")
        print(f"| :--- | :--- | :--- | :--- | :--- | :--- |")
        for name, summary in results.items():
            acc = f"{summary['accuracy_mean']:.4f} ± {summary['accuracy_std']:.4f}"
            kappa = f"{summary['cohen_kappa_mean']:.4f} ± {summary['cohen_kappa_std']:.4f}"
            sens = f"{summary['sensitivity_mean']:.4f} ± {summary['sensitivity_std']:.4f}"
            spec = f"{summary['specificity_mean']:.4f} ± {summary['specificity_std']:.4f}"
            auc_val = f"{summary['auc_mean']:.4f} ± {summary['auc_std']:.4f}"
            print(f"| {name} | {acc} | {kappa} | {sens} | {spec} | {auc_val} |")
    else:
        print(f"| 模型名稱 | Accuracy (Mean ± Std) | Cohen's Kappa (Mean ± Std) | Macro Sens (Mean ± Std) | Macro Spec (Mean ± Std) | AUC-ROC (Mean ± Std) |")
        print(f"| :--- | :--- | :--- | :--- | :--- | :--- |")
        for name, summary in results.items():
            acc = f"{summary['accuracy_mean']:.4f} ± {summary['accuracy_std']:.4f}"
            kappa = f"{summary['cohen_kappa_mean']:.4f} ± {summary['cohen_kappa_std']:.4f}"
            sens = f"{summary['macro_sensitivity_mean']:.4f} ± {summary['macro_sensitivity_std']:.4f}"
            spec = f"{summary['macro_specificity_mean']:.4f} ± {summary['macro_specificity_std']:.4f}"
            auc_val = f"{summary['auc_mean']:.4f} ± {summary['auc_std']:.4f}"
            print(f"| {name} | {acc} | {kappa} | {sens} | {spec} | {auc_val} |")
            
    print("=" * 80 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="執行多模型睡眠階段預測評估流程",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default=str(PROCESSED_DIR),
        help="特徵工程產出的 Parquet 目錄",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=20,
        help="Monte Carlo 交叉驗證的拆分次數",
    )
    parser.add_argument(
        "--train-size",
        type=float,
        default=0.7,
        help="每次拆分中訓練集受試者的比例",
    )
    parser.add_argument(
        "--n-classes",
        type=int,
        choices=[2, 3],
        default=2,
        help="預測的睡眠階段類別數 (2: Wake/Sleep, 3: Wake/NREM/REM)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="並行處理使用的 CPU 核心數 (-1 表示使用全部核心)",
    )

    args = parser.parse_args()

    run_evaluation(
        processed_dir=Path(args.processed_dir),
        n_splits=args.n_splits,
        train_size=args.train_size,
        n_classes=args.n_classes,
        n_jobs=args.n_jobs,
    )


if __name__ == "__main__":
    main()
