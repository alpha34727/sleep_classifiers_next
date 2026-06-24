import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)

# 加入專案路徑以確保 import 可用
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


class SleepWakeAnalyzer:
    """睡眠/清醒分類評估與圖表繪製核心類別 (Object-Oriented)"""

    def __init__(
        self,
        processed_dir: Union[str, Path] = PROCESSED_DIR,
        output_dir: Union[str, Path] = PROJECT_ROOT / "outputs",
        n_splits: int = 20,
        train_size: float = 0.7,
        n_jobs: int = -1,
    ):
        self.processed_dir = Path(processed_dir)
        self.output_dir = Path(output_dir)
        self.n_splits = n_splits
        self.train_size = train_size
        self.n_jobs = n_jobs

        # 緩存特徵數據
        self.X: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None
        self.groups: Optional[np.ndarray] = None

        # 4 種特徵組合與對應的欄位索引 (count, hr, time, cosine)
        self.feature_sets = {
            "Motion only": [0],
            "HR only": [1],
            "Motion, HR": [0, 1],
            "Motion, HR, and Cosine": [0, 1, 2, 3],  # Cosine 組合包含所有特徵 (含 time 與 cosine)
        }

        # 對齊論文原圖的配色方案 (Aesthetics)
        self.feature_colors = {
            "Motion only": "#2b5c8f",              # 鋼鐵藍
            "HR only": "#ffa500",                  # 橘黃
            "Motion, HR": "#2ca02c",               # 翠綠
            "Motion, HR, and Cosine": "#59174b",   # 深紫/洋紅
        }

    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        載入並快取所有受試者特徵數據至記憶體 (In-Memory Caching)
        """
        if self.X is not None and self.y is not None:
            return self.X, self.y, self.groups

        # 掃描 Processed 目錄下所有受試者 parquet
        subject_ids = sorted([p.stem for p in self.processed_dir.glob("*.parquet")])
        if not subject_ids:
            raise ValueError(
                f"在目錄 {self.processed_dir} 下找不到任何 .parquet 檔案，請確認特徵工程管線已執行。"
            )

        print(f"[*] 正在載入所有受試者數據 (共 {len(subject_ids)} 位受試者)...")
        self.X, self.y, self.groups = DataLoader.load_all_subjects(
            processed_dir=self.processed_dir,
            subject_ids=subject_ids,
            n_classes=2,
        )
        print(f"[*] 載入完成。特徵維度: {self.X.shape}, 樣本數: {len(self.y)}")
        return self.X, self.y, self.groups

    def evaluate_models(
        self,
        model_classes: Dict[str, Type[BaseSleepClassifier]],
    ) -> Dict[str, Dict[str, Dict[str, list]]]:
        """
        針對每個模型、每種特徵組合，執行 Monte Carlo 交叉驗證
        """
        X, y, groups = self.load_data()
        results = {}

        for model_name, model_cls in model_classes.items():
            results[model_name] = {}
            print(f"\n[*] 正在執行模型評估: {model_name}...")
            
            for feat_name, col_indices in self.feature_sets.items():
                print(f"    -> 評估特徵組合: {feat_name} ({self.n_splits} Splits)...")
                # 擷取特徵子集
                X_subset = X[:, col_indices]

                mccv_res = SleepValidator.run_mccv(
                    model_cls=model_cls,
                    X=X_subset,
                    y=y,
                    groups=groups,
                    n_splits=self.n_splits,
                    train_size=self.train_size,
                    n_classes=2,
                    n_jobs=self.n_jobs,
                )

                results[model_name][feat_name] = {
                    "all_y_test": mccv_res["all_y_test"],
                    "all_y_prob": mccv_res["all_y_prob"],
                }

        return results

    def plot_roc_curves(
        self,
        results: Dict[str, Dict[str, Dict[str, list]]],
        save_path: Path,
    ) -> None:
        """
        繪製 Figure 2：ROC 曲線多面板圖 (3x2 網格)
        每個面板代表一個演算法，顏色代表不同特徵組合，並附帶標準差陰影
        """
        sns.set_theme(style="ticks")
        
        # 建立 4x2 畫布
        fig, axes = plt.subplots(4, 2, figsize=(15, 25))
        
        # 設定字體樣式
        plt.rcParams.update({
            "font.size": 13,
            "axes.labelsize": 15,
            "axes.titlesize": 17,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 11,
        })

        model_names = list(results.keys())
        mean_fpr = np.linspace(0, 1, 100)

        # 逐一繪製子圖
        for i, name in enumerate(model_names):
            row = i // 2
            col = i % 2
            ax = axes[row, col]

            ax.set_title(name, fontsize=18, fontweight="bold", pad=12)

            # 針對該模型的 4 種特徵組合畫線
            for feat_name, data in results[name].items():
                tprs = []
                aucs = []
                color = self.feature_colors.get(feat_name, "#333333")

                for y_true, y_prob in zip(data["all_y_test"], data["all_y_prob"]):
                    if y_prob.ndim == 2:
                        y_prob = y_prob[:, 1]

                    fpr, tpr, _ = roc_curve(y_true, y_prob)
                    interp_tpr = np.interp(mean_fpr, fpr, tpr)
                    interp_tpr[0] = 0.0
                    tprs.append(interp_tpr)
                    aucs.append(auc(fpr, tpr))

                mean_tpr = np.mean(tprs, axis=0)
                mean_tpr[-1] = 1.0
                std_tpr = np.std(tprs, axis=0)
                mean_auc = np.mean(aucs)
                std_auc = np.std(aucs)

                # 畫平均線與陰影帶
                ax.plot(
                    mean_fpr,
                    mean_tpr,
                    color=color,
                    lw=2.2,
                    label=f"{feat_name} (AUC = {mean_auc:.3f} ± {std_auc:.3f})",
                )
                tpr_upper = np.clip(mean_tpr + std_tpr, 0, 1)
                tpr_lower = np.clip(mean_tpr - std_tpr, 0, 1)
                ax.fill_between(
                    mean_fpr,
                    tpr_lower,
                    tpr_upper,
                    color=color,
                    alpha=0.12,
                )

            # 對角機會線
            ax.plot([0, 1], [0, 1], color="gray", lw=1.2, linestyle="--", label="Chance")
            
            # 設定座標軸樣式與範疇
            ax.set_xlim([-0.02, 1.02])
            ax.set_ylim([-0.02, 1.02])
            ax.set_xlabel("Fraction of wake scored as sleep", fontsize=14)
            ax.set_ylabel("Fraction of sleep scored as sleep", fontsize=14)
            ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="none")
            sns.despine(ax=ax) # 移除 top/right 外框

        # 處理剩餘的第 7 與第 8 個面板，第 7 面板繪製一個全域共享的說明 Legend 面板，第 8 面板隱藏
        ax_empty = axes[3, 0]
        ax_empty.axis("off") # 隱藏空白子圖座標軸
        
        ax_hidden = axes[3, 1]
        ax_hidden.axis("off") # 隱藏第 8 個空白子圖
        
        # 繪製美化的說明框
        legend_handles, legend_labels = axes[0, 0].get_legend_handles_labels()
        # 移除 Chance 線避免重複
        clean_handles = [h for h, l in zip(legend_handles, legend_labels) if l != "Chance"]
        clean_labels = [l.split(" (")[0] for l in legend_labels if l != "Chance"]
        
        ax_empty.legend(
            clean_handles,
            clean_labels,
            loc="center",
            fontsize=15,
            frameon=True,
            facecolor="white",
            shadow=False,
            title="Feature Sets",
            title_fontsize=16,
        )

        fig.suptitle(
            "Figure 2: ROC Curves Across Multiple Classifiers and Feature Configurations",
            fontsize=20,
            fontweight="bold",
            y=0.98,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # 儲存
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"[+] ROC 網格子圖已儲存至: {save_path}")

    def plot_pr_curves(
        self,
        results: Dict[str, Dict[str, Dict[str, list]]],
        save_path: Path,
    ) -> None:
        """
        繪製 Figure 3：Precision-Recall 曲線多面板圖 (3x2 網格)
        以 Wake=0 為正分類
        """
        sns.set_theme(style="ticks")
        
        # 建立 4x2 畫布
        fig, axes = plt.subplots(4, 2, figsize=(15, 25))
        
        # 設定字體樣式
        plt.rcParams.update({
            "font.size": 13,
            "axes.labelsize": 15,
            "axes.titlesize": 17,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 11,
        })

        model_names = list(results.keys())
        mean_recall = np.linspace(0, 1, 100)

        # 逐一繪製子圖
        for i, name in enumerate(model_names):
            row = i // 2
            col = i % 2
            ax = axes[row, col]

            ax.set_title(name, fontsize=18, fontweight="bold", pad=12)

            # 針對該模型的 4 種特徵組合畫 PR 線
            for feat_name, data in results[name].items():
                precisions = []
                aps = []
                color = self.feature_colors.get(feat_name, "#333333")

                for y_true, y_prob in zip(data["all_y_test"], data["all_y_prob"]):
                    if y_prob.ndim == 2:
                        y_prob = y_prob[:, 1]

                    # 以 Wake 為正分類
                    y_true_wake = (y_true == 0).astype(int)
                    y_prob_wake = 1.0 - y_prob

                    precision, recall, _ = precision_recall_curve(y_true_wake, y_prob_wake)

                    # 反轉以供插值
                    recall_reversed = recall[::-1]
                    precision_reversed = precision[::-1]

                    interp_precision = np.interp(mean_recall, recall_reversed, precision_reversed)
                    precisions.append(interp_precision)
                    aps.append(average_precision_score(y_true_wake, y_prob_wake))

                mean_precision = np.mean(precisions, axis=0)
                std_precision = np.std(precisions, axis=0)
                mean_ap = np.mean(aps)
                std_ap = np.std(aps)

                # 畫平均線與標準差陰影
                ax.plot(
                    mean_recall,
                    mean_precision,
                    color=color,
                    lw=2.2,
                    label=f"{feat_name} (AP = {mean_ap:.3f} ± {std_ap:.3f})",
                )
                prec_upper = np.clip(mean_precision + std_precision, 0, 1)
                prec_lower = np.clip(mean_precision - std_precision, 0, 1)
                ax.fill_between(
                    mean_recall,
                    prec_lower,
                    prec_upper,
                    color=color,
                    alpha=0.12,
                )

            # 繪製隨機機會水平線 (Wake 的 Prevalence)
            if self.y is not None:
                wake_prevalence = np.mean(self.y == 0)
                ax.axhline(
                    y=wake_prevalence,
                    color="gray",
                    lw=1.2,
                    linestyle="--",
                    label=f"Chance ({wake_prevalence:.3f})",
                )

            ax.set_xlim([-0.02, 1.02])
            ax.set_ylim([-0.02, 1.02])
            ax.set_xlabel("Fraction of wake scored as wake", fontsize=14)
            ax.set_ylabel("Fraction of predicted wake correct", fontsize=14)
            ax.legend(loc="lower left", frameon=True, facecolor="white", edgecolor="none")
            sns.despine(ax=ax)

        # 處理剩餘的第 7 與第 8 個面板，第 7 面板繪製一個全域共享的說明 Legend 面板，第 8 面板隱藏
        ax_empty = axes[3, 0]
        ax_empty.axis("off") # 隱藏空白子圖座標軸
        
        ax_hidden = axes[3, 1]
        ax_hidden.axis("off") # 隱藏第 8 個空白子圖
        
        # 取得 Legend 控制器
        legend_handles, legend_labels = axes[0, 0].get_legend_handles_labels()
        clean_handles = [h for h, l in zip(legend_handles, legend_labels) if "Chance" not in l]
        clean_labels = [l.split(" (")[0] for l in legend_labels if "Chance" not in l]

        ax_empty.legend(
            clean_handles,
            clean_labels,
            loc="center",
            fontsize=15,
            frameon=True,
            facecolor="white",
            shadow=False,
            title="Feature Sets",
            title_fontsize=16,
        )

        fig.suptitle(
            "Figure 3: Precision-Recall Curves Across Multiple Classifiers and Feature Configurations",
            fontsize=20,
            fontweight="bold",
            y=0.98,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # 儲存
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"[+] PR 網格子圖已儲存至: {save_path}")


def figures_mc_sleep_wake(
    processed_dir: Union[str, Path] = PROCESSED_DIR,
    output_dir: Union[str, Path] = PROJECT_ROOT / "outputs",
    n_splits: int = 20,
    train_size: float = 0.7,
    n_jobs: int = -1,
) -> None:
    """
    主要評估與繪圖進入點函式。對齊原著論文的 Figure 2 與 Figure 3 生成。
    """
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 定義要評估並列出子圖的所有分類器模型
    model_classes = {
        "Random Forest": RandomForestClassifierWrapper,
        "Logistic Regression": LogisticRegressionClassifier,
        "Neural Net": MLPClassifierWrapper,
        "LightGBM": LightGBMClassifier,
        "XGBoost": XGBoostClassifier,
        "k-Nearest Neighbors": KNNClassifierWrapper,
    }

    # 初始化評估器
    analyzer = SleepWakeAnalyzer(
        processed_dir=processed_dir,
        output_dir=output_dir,
        n_splits=n_splits,
        train_size=train_size,
        n_jobs=n_jobs,
    )

    # 載入所有受試者數據
    analyzer.load_data()

    # 執行 Monte Carlo 多模型、多特徵集合評估
    results = analyzer.evaluate_models(model_classes)

    # 繪製並儲存 Figure 2 與 Figure 3 網格子圖
    analyzer.plot_roc_curves(results, save_path=figures_dir / "figure2_roc.png")
    analyzer.plot_pr_curves(results, save_path=figures_dir / "figure3_pr.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="執行 Monte Carlo 交叉驗證並繪製各模型在不同特徵組合下的 ROC (Figure 2) 與 PR (Figure 3) 網格曲線圖。"
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default=str(PROCESSED_DIR),
        help="Processed 特徵 Parquet 檔案目錄",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "outputs"),
        help="輸出目錄 (包含 figures/ 子目錄)",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=20,
        help="Monte Carlo 拆分次數 (預設: 20)",
    )
    parser.add_argument(
        "--train-size",
        type=float,
        default=0.7,
        help="訓練受試者佔比 (預設: 0.7)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="並行處理核心數 (預設: -1，使用全部 CPU 核心)",
    )

    args = parser.parse_args()

    print("[*] 開始執行 Sleep/Wake 多模型多特徵評估與圖表繪製流程...")
    figures_mc_sleep_wake(
        processed_dir=args.processed_dir,
        output_dir=args.output_dir,
        n_splits=args.n_splits,
        train_size=args.train_size,
        n_jobs=args.n_jobs,
    )
    print("[+] 評估與繪圖工作全部完成！")
