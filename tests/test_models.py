import sys
import tempfile
from pathlib import Path
import numpy as np
import polars as pl
import pytest

# 將專案根目錄加入 sys.path 以防 ModuleNotFoundError
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data_loader import DataLoader
from src.config import FEATURE_COLS
from src.models.baseline import (
    LogisticRegressionClassifier,
    RandomForestClassifierWrapper,
    MLPClassifierWrapper,
    KNNClassifierWrapper
)
from src.models.modern_gbdt import LightGBMClassifier, XGBoostClassifier

# 所有模型列表
MODELS = [
    LogisticRegressionClassifier,
    RandomForestClassifierWrapper,
    MLPClassifierWrapper,
    LightGBMClassifier,
    XGBoostClassifier,
    KNNClassifierWrapper
]

def test_data_loader_label_mapping():
    """驗證 DataLoader 對二分類與三分類睡眠標籤的映射正確性"""
    # 建立模擬的受試者 Parquet 數據
    subject_id = "test_subj"
    mock_data = pl.DataFrame({
        "timestamp": [100.0, 130.0, 160.0, 190.0, 220.0, 250.0],
        # 睡眠階段標籤：0=Wake, 1=N1, 2=N2, 3=N3, 4=N4, 5=REM, -1=Unscored
        "stage": [0, 1, 2, 5, -1, 3],
        "count_feature": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        "hr_feature": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "time_feature": [0.0, 0.01, 0.02, 0.03, 0.04, 0.05],
        "cosine_feature": [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
    })
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        mock_data.write_parquet(tmp_path / f"{subject_id}.parquet")
        
        # 1. 測試二分類映射
        X2, y2, groups2 = DataLoader.load_all_subjects(tmp_path, [subject_id], n_classes=2)
        # -1 (Unscored) 會被過濾掉，所以剩餘 5 個點
        assert X2.shape == (5, 4)
        assert len(y2) == 5
        # y2 應該只有 0 (Wake) 與 1 (Sleep)
        assert np.array_equal(y2, np.array([0, 1, 1, 1, 1]))
        assert np.array_equal(groups2, np.array([subject_id] * 5))
        
        # 2. 測試三分類映射
        X3, y3, groups3 = DataLoader.load_all_subjects(tmp_path, [subject_id], n_classes=3)
        assert X3.shape == (5, 4)
        assert len(y3) == 5
        # y3 應該是 0 (Wake), 1 (NREM: N1/N2/N3), 2 (REM: 5)
        # stage 映射: 0->0, 1->1, 2->1, 5->2, 3->1
        assert np.array_equal(y3, np.array([0, 1, 1, 2, 1]))
        assert np.array_equal(groups3, np.array([subject_id] * 5))

@pytest.mark.parametrize("model_cls", MODELS)
@pytest.mark.parametrize("n_classes", [2, 3])
def test_models_fit_predict_save(model_cls, n_classes):
    """驗證所有模型在二分類與三分類下具有相同的 API 接口，且 fit, predict 正常"""
    # 建立模擬訓練數據
    np.random.seed(42)
    X = np.random.rand(50, 4)
    if n_classes == 2:
        y = np.random.randint(0, 2, size=50)
    else:
        y = np.random.randint(0, 3, size=50)
        
    model = model_cls()
    
    # 1. 訓練
    model.fit(X, y)
    
    # 2. 預測
    preds = model.predict(X)
    assert preds.shape == (50,)
    assert set(preds).issubset(set(y))
    
    # 3. 概率預測
    probs = model.predict_proba(X)
    assert probs.shape == (50, n_classes)
    np.testing.assert_allclose(np.sum(probs, axis=1), 1.0, atol=1e-5)
    
    # 4. 儲存與載入
    with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
        model_path = Path(tmpfile.name)
    try:
        model.save_model(model_path)
        
        # 重新建立一個新實例並載入
        new_model = model_cls()
        new_model.load_model(model_path)
        
        # 驗證載入後的預測結果是否與原先一致
        new_preds = new_model.predict(X)
        assert np.array_equal(preds, new_preds)
    finally:
        if model_path.is_file():
            model_path.unlink()
