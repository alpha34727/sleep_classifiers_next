import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
from typing import List, Optional, Union
from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score, confusion_matrix

class SleepPlotter:
    @staticmethod
    def plot_roc_curve(y_true: np.ndarray, 
                      y_prob: np.ndarray, 
                      n_classes: int = 2, 
                      class_names: Optional[List[str]] = None,
                      title: str = "Receiver Operating Characteristic (ROC) Curve", 
                      save_path: Optional[Union[str, Path]] = None) -> None:
        """
        繪製 ROC 曲線與計算 AUC。支援二分類與多分類（OVR）。
        """
        plt.figure(figsize=(8, 6))
        sns.set_theme(style="whitegrid")
        
        if n_classes == 2:
            # 取得正類（class 1）機率
            prob_positive = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
            fpr, tpr, _ = roc_curve(y_true, prob_positive)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, color="darkorange", lw=2, 
                     label=f"ROC Curve (AUC = {roc_auc:.3f})")
        else:
            # 多分類：為每個類別繪製獨立的 ROC 曲線 (One-Vs-Rest)
            if class_names is None:
                class_names = [f"Class {i}" for i in range(n_classes)]
                
            fpr_dict = {}
            tpr_dict = {}
            roc_auc_dict = {}
            
            for c in range(n_classes):
                y_true_c = (y_true == c).astype(int)
                y_prob_c = y_prob[:, c]
                fpr_dict[c], tpr_dict[c], _ = roc_curve(y_true_c, y_prob_c)
                roc_auc_dict[c] = auc(fpr_dict[c], tpr_dict[c])
                
                plt.plot(fpr_dict[c], tpr_dict[c], lw=2,
                         label=f"ROC of {class_names[c]} (AUC = {roc_auc_dict[c]:.3f})")
                
            # 計算 Macro-average ROC
            all_fpr = np.unique(np.concatenate([fpr_dict[c] for c in range(n_classes)]))
            mean_tpr = np.zeros_like(all_fpr)
            for c in range(n_classes):
                mean_tpr += np.interp(all_fpr, fpr_dict[c], tpr_dict[c])
            mean_tpr /= n_classes
            macro_auc = auc(all_fpr, mean_tpr)
            
            plt.plot(all_fpr, mean_tpr, color="navy", linestyle=":", lw=3,
                     label=f"Macro-average ROC (AUC = {macro_auc:.3f})")

        plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate (FPR)")
        plt.ylabel("True Positive Rate (TPR)")
        plt.title(title, fontsize=14, fontweight="bold", pad=15)
        plt.legend(loc="lower right", frameon=True)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.close()

    @staticmethod
    def plot_pr_curve(y_true: np.ndarray, 
                     y_prob: np.ndarray, 
                     n_classes: int = 2, 
                     class_names: Optional[List[str]] = None,
                     title: str = "Precision-Recall Curve", 
                     save_path: Optional[Union[str, Path]] = None) -> None:
        """
        繪製 Precision-Recall 曲線與計算 Average Precision (AP)。支援二分類與多分類（OVR）。
        """
        plt.figure(figsize=(8, 6))
        sns.set_theme(style="whitegrid")
        
        if n_classes == 2:
            prob_positive = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
            precision, recall, _ = precision_recall_curve(y_true, prob_positive)
            ap = average_precision_score(y_true, prob_positive)
            
            plt.plot(recall, precision, color="teal", lw=2, 
                     label=f"PR Curve (AP = {ap:.3f})")
        else:
            if class_names is None:
                class_names = [f"Class {i}" for i in range(n_classes)]
                
            for c in range(n_classes):
                y_true_c = (y_true == c).astype(int)
                y_prob_c = y_prob[:, c]
                precision, recall, _ = precision_recall_curve(y_true_c, y_prob_c)
                ap_c = average_precision_score(y_true_c, y_prob_c)
                
                plt.plot(recall, precision, lw=2,
                         label=f"PR of {class_names[c]} (AP = {ap_c:.3f})")

        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.title(title, fontsize=14, fontweight="bold", pad=15)
        plt.legend(loc="lower left", frameon=True)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.close()

    @staticmethod
    def plot_bland_altman(y_true_duration: np.ndarray, 
                          y_pred_duration: np.ndarray, 
                          title: str = "Bland-Altman Plot (PSG vs Predicted Sleep Time)", 
                          save_path: Optional[Union[str, Path]] = None) -> None:
        """
        繪製 Bland-Altman 一致性分析偏差圖。
        y_true_duration: 真實睡眠時長 (例如單位為分鐘或小時)
        y_pred_duration: 模型預測睡眠時長
        """
        diffs = y_true_duration - y_pred_duration
        means = (y_true_duration + y_pred_duration) / 2.0
        
        bias = np.mean(diffs)
        sd = np.std(diffs)
        upper_loa = bias + 1.96 * sd
        lower_loa = bias - 1.96 * sd
        
        plt.figure(figsize=(8, 6))
        sns.set_theme(style="whitegrid")
        
        # 繪製散點
        plt.scatter(means, diffs, alpha=0.6, color="purple", edgecolors="none", s=50)
        
        # 繪製 Bias 與 LOA 橫線
        plt.axhline(bias, color="red", linestyle="--", lw=2, 
                    label=f"Mean Bias ({bias:+.2f})")
        plt.axhline(upper_loa, color="royalblue", linestyle=":", lw=2, 
                    label=f"+1.96 SD ({upper_loa:+.2f})")
        plt.axhline(lower_loa, color="royalblue", linestyle=":", lw=2, 
                    label=f"-1.96 SD ({lower_loa:+.2f})")
        
        # 填充 LOA 區間
        plt.axhspan(lower_loa, upper_loa, color="royalblue", alpha=0.1)
        
        plt.xlabel("Mean of PSG and Predicted (units)")
        plt.ylabel("Difference (PSG - Predicted)")
        plt.title(title, fontsize=14, fontweight="bold", pad=15)
        plt.legend(loc="best", frameon=True)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.close()

    @staticmethod
    def plot_confusion_matrix(y_true: np.ndarray, 
                              y_pred: np.ndarray, 
                              class_names: Optional[List[str]] = None, 
                              title: str = "Normalized Confusion Matrix", 
                              save_path: Optional[Union[str, Path]] = None) -> None:
        """
        繪製歸一化後的混淆矩陣 Heatmap。
        """
        cm = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(6, 5))
        
        if class_names is None:
            n_classes = len(np.unique(y_true))
            class_names = [f"Class {i}" for i in range(n_classes)]
            
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", 
                    xticklabels=class_names, yticklabels=class_names, 
                    cbar=True, square=True)
        
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.title(title, fontsize=14, fontweight="bold", pad=15)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.close()
