import sys
from pathlib import Path
import numpy as np
import pytest

# 將專案根目錄加入 sys.path 以防 ModuleNotFoundError
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.evaluation.validator import SleepValidator
from src.models.baseline import LogisticRegressionClassifier
from src.models.modern_gbdt import LightGBMClassifier

def test_calculate_metrics_binary():
    """測試二分類的指標計算"""
    y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 1, 0, 1, 0, 1, 0])
    # 模擬概率
    y_prob = np.array([
        [0.8, 0.2],
        [0.3, 0.7],
        [0.1, 0.9],
        [0.6, 0.4],
        [0.2, 0.8],
        [0.9, 0.1],
        [0.3, 0.7],
        [0.7, 0.3]
    ])
    
    metrics = SleepValidator.calculate_metrics(y_true, y_pred, y_prob, n_classes=2)
    
    assert "accuracy" in metrics
    assert "cohen_kappa" in metrics
    assert "auc" in metrics
    assert "sensitivity" in metrics
    assert "specificity" in metrics
    
    # 數值應介於 0 與 1 之間
    for k, v in metrics.items():
        assert 0.0 <= v <= 1.0, f"Metric {k} out of bounds: {v}"

def test_calculate_metrics_multiclass():
    """測試三分類的指標計算"""
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1])
    y_pred = np.array([0, 1, 2, 1, 1, 0, 0, 2])
    y_prob = np.random.rand(8, 3)
    # 將機率歸一化
    y_prob = y_prob / np.sum(y_prob, axis=1, keepdims=True)
    
    metrics = SleepValidator.calculate_metrics(y_true, y_pred, y_prob, n_classes=3)
    
    assert "accuracy" in metrics
    assert "cohen_kappa" in metrics
    assert "auc" in metrics
    assert "macro_sensitivity" in metrics
    assert "macro_specificity" in metrics
    
    for c in range(3):
        assert f"sensitivity_class_{c}" in metrics
        assert f"specificity_class_{c}" in metrics
        
    for k, v in metrics.items():
        assert 0.0 <= v <= 1.0, f"Metric {k} out of bounds: {v}"

def test_validation_workflows():
    """測試整個 LOOCV 與 MCCV 的交叉驗證流程"""
    np.random.seed(42)
    # 建立 5 個受試者的模擬數據，每個受試者有 20 個樣本
    n_samples = 100
    X = np.random.rand(n_samples, 4)
    y = np.random.randint(0, 2, size=n_samples)
    
    # 產生群組標識：[subj1]*20, [subj2]*20 ...
    groups = np.array([f"subj_{i//20 + 1}" for i in range(n_samples)])
    
    # 1. 測試 LOOCV
    loocv_results = SleepValidator.run_loocv(
        model_cls=LogisticRegressionClassifier,
        X=X, y=y, groups=groups, n_classes=2
    )
    
    assert "global_metrics" in loocv_results
    assert "subject_summary" in loocv_results
    assert "raw_subject_metrics" in loocv_results
    assert len(loocv_results["raw_subject_metrics"]) == 5 # 5 個受試者
    
    # 檢查 summary 中是否包含 Mean 與 Std
    assert "accuracy_mean" in loocv_results["subject_summary"]
    assert "accuracy_std" in loocv_results["subject_summary"]
    
    # 2. 測試 MCCV (GBDT 模型如 LightGBM)
    mccv_results = SleepValidator.run_mccv(
        model_cls=LightGBMClassifier,
        X=X, y=y, groups=groups,
        n_splits=5, train_size=0.6, n_classes=2,
        n_estimators=10 # 使用少量的 trees 以加速測試
    )
    
    assert "mccv_summary" in mccv_results
    assert "raw_trial_metrics" in mccv_results
    assert len(mccv_results["raw_trial_metrics"]) == 5 # 5 splits
    assert "accuracy_mean" in mccv_results["mccv_summary"]
    assert "accuracy_std" in mccv_results["mccv_summary"]

def test_sleep_plotter():
    """測試 SleepPlotter 繪圖方法是否能正常生成圖片且無語法錯誤"""
    import tempfile
    from src.evaluation.plotter import SleepPlotter
    
    # 建立模擬數據
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    y_pred = np.array([0, 1, 1, 0, 2, 2, 0, 1, 1, 0])
    y_prob = np.random.rand(10, 3)
    y_prob = y_prob / np.sum(y_prob, axis=1, keepdims=True)
    
    true_duration = np.array([480.0, 500.0, 450.0, 420.0, 510.0])
    pred_duration = np.array([475.0, 512.0, 440.0, 430.0, 498.0])
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        # 1. 測試二分類 ROC 與 PR 繪圖
        roc2_path = tmp_path / "roc2.png"
        pr2_path = tmp_path / "pr2.png"
        SleepPlotter.plot_roc_curve(y_true[:5] % 2, y_prob[:5, :2], n_classes=2, save_path=roc2_path)
        SleepPlotter.plot_pr_curve(y_true[:5] % 2, y_prob[:5, :2], n_classes=2, save_path=pr2_path)
        assert roc2_path.is_file()
        assert pr2_path.is_file()
        
        # 2. 測試三分類 ROC 與 PR 繪圖
        roc3_path = tmp_path / "roc3.png"
        pr3_path = tmp_path / "pr3.png"
        class_names = ["Wake", "NREM", "REM"]
        SleepPlotter.plot_roc_curve(y_true, y_prob, n_classes=3, class_names=class_names, save_path=roc3_path)
        SleepPlotter.plot_pr_curve(y_true, y_prob, n_classes=3, class_names=class_names, save_path=pr3_path)
        assert roc3_path.is_file()
        assert pr3_path.is_file()
        
        # 3. 測試 Bland-Altman 繪圖
        ba_path = tmp_path / "bland_altman.png"
        SleepPlotter.plot_bland_altman(true_duration, pred_duration, save_path=ba_path)
        assert ba_path.is_file()
        
        # 4. 測試混淆矩陣繪圖
        cm_path = tmp_path / "cm.png"
        SleepPlotter.plot_confusion_matrix(y_true, y_pred, class_names=class_names, save_path=cm_path)
        assert cm_path.is_file()

