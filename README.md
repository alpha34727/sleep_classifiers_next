# Sleep Classifiers Next：Apple Watch 與穿戴式裝置睡眠階段預測重構專案

本專案是針對 Walch 等人發表於《SLEEP》（2019年）的經典論文——*「Sleep stage prediction with raw acceleration and photoplethysmography heart rate data derived from a consumer wearable device」*（Walch et al., SLEEP 2019）所開發的現代化、高效率且具備生產力等級的 Python 3.12 重構版本。

本專案旨在解決原學術程式碼記憶體消耗過大（處理單一受試者需耗費 150GB+ 記憶體）的問題，透過全新設計的「硬碟快取 ETL 引擎」與「策略模式分類器架構」，在記憶體消耗限制於 **4GB RAM 以下** 的前提下，完全復現論文中所有的圖表與預測指標。

---

## 1. 專案概述與重構亮點

### 舊版學術程式碼 vs. 現代化重構版本對比

| 評估維度 | 舊版學術程式碼 (Legacy) | 現代化重構版本 (Next) |
| :--- | :--- | :--- |
| **Python 版本** | Python 3.7+ (面臨套件相容性斷代) | **Python 3.12** (享有現代語法與效能最佳化) |
| **套件管理器** | pip / conda (依賴關係緩慢且易衝突) | **uv** (基於 Rust 編寫，極速同步與建置環境) |
| **記憶體消耗** | **150GB+ RAM** (一次性載入大量全量 Pandas 矩陣) | **< 4GB RAM** (基於硬碟快取，逐一主體串流 Parquet 運算) |
| **資料儲存格式** | 多個中間 `.out` 文字檔 (讀寫慢且遺漏首行 Header) | 精簡、壓縮的 **Apache Parquet** 欄位式儲存檔 |
| **核心演算法設計** | 用 `np.where` 與雙重迴圈遍歷，複雜度達 $O(N)$ | 正則化網格下的 **$O(1)$ 切片索引**，運算加速百倍 |
| **模型架構擴充** | 寫死於 Sklearn 類別中，難以整合新演算法 | **策略模式 (Strategy Pattern)**，隨插即用 LightGBM/XGBoost |
| **穩健性測試** | 依賴第三方 Mock 套件，且存在時區敏感的 Bug | 使用標準 `unittest` 整合等價性驗證，時區感知且跨平台 |

---

## 2. 環境配置與快速上手

本專案使用 `uv` 作為專案與套件管理工具。請依據以下步驟配置執行環境：

### 2.1 安裝 `uv`
若尚未安裝 `uv`，請在終端機執行以下指令：
```powershell
# Windows PowerShell 安裝指令
powershell -c "irts https://astral.sh/uv/install.ps1 | iex"
```

### 2.2 複製專案並同步依賴關係
```bash
# 複製儲存庫
git clone https://github.com/alpha34727/sleep_classifiers_next.git
cd sleep_classifiers_next

# 建立虛擬環境並安裝所有依賴套件（包含開發測試依賴）
uv sync
```

### 2.3 原始資料集放置規範
請將 Apple Watch 與 MESA 生理感測資料放置於專案根目錄的 `data/` 資料夾下，結構如下：
```text
data/
├── labels/         # 包含受試者 PSG 黃金標準標籤，如 {subject_id}_labeled_sleep.txt
├── motion/         # 包含 raw MEMS 加速度資料，如 {subject_id}_acceleration.txt
├── heart_rate/     # 包含 PPG 心率資料，如 {subject_id}_heartrate.txt
└── steps/          # 包含計步感測資料，如 {subject_id}_steps.txt
```

---

## 3. 端到端復現工作流

重構後的 CLI 介面整合在單一入口指令 `sleep-classify` 中，可透過 `uv run` 直接調用。

### 步驟一：硬碟快取與特徵 ETL 生成
將原始的加速度、心率與計步資料進行對齊、重採樣、濾波（Butterworth 帶通濾波器與高斯差分 DoG 濾波器），並生成 30秒 Epoch 區段的特徵矩陣，直接存成 `.parquet` 檔案：
```bash
# 執行逐一主體的硬碟快取特徵提取
uv run sleep-classify preprocess --data-dir ./data --output-dir ./outputs
```
*(注意：此步驟會在執行時主動調用 Garbage Collection `gc.collect()` 以釋放記憶體，記憶體佔用峰值 < 4GB)*

### 步驟二：執行對齊度驗證與基準測試
在「主體留一交叉驗證 (Leave-One-Subject-Out, LOSO CV)」協定下，評估隨機森林、邏輯斯迴歸、k-Nearest Neighbors、MLP 神經網路與 LightGBM 在各特徵組合下的表現，並輸出對齊論文 Table 2 至 Table 9 的預測報表：
```bash
# 執行基準驗證並印出與論文數據對齊的 Tables 報表
uv run sleep-classify benchmark --data-dir ./data --output-dir ./outputs
```

---

## 4. 論文圖表一鍵生成指南

透過 `plot` 指令可以復現 Walch et al. (2019) 論文中所有的視覺化圖表，生成的結果會自動儲存於 `outputs/figures/` 目錄下：

| 論文對應圖表 | 生成指令 (CLI Command) | 輸出檔案位置 | 圖表物理意義說明 |
| :--- | :--- | :--- | :--- |
| **Figure 1** | `uv run sleep-classify plot --fig 1` | `outputs/figures/fig1_sample_night.png` | 單一受試者整夜動態、心率與睡眠階段生理時序圖 |
| **Figure 2** | `uv run sleep-classify plot --fig 2` | `outputs/figures/fig2_roc_sleep_wake.png` | 區分 Sleep/Wake 的各分類器 ROC 曲線比較 |
| **Figure 3** | `uv run sleep-classify plot --fig 3` | `outputs/figures/fig3_pr_sleep_wake.png` | 針對 Wake（少數類別）分類的精確度-召回率曲線 |
| **Figure 4** | `uv run sleep-classify plot --fig 4` | `outputs/figures/fig4_roc_staging.png` | 三階段 (Wake/NREM/REM) 分類之 ROC 曲線 |
| **Figure 5** | `uv run sleep-classify plot --fig 5` | `outputs/figures/fig5_bland_altman.png` | 預測 TST 等睡眠指標的 Bland-Altman 一致性分析 |
| **Figure 6 & 7**| `uv run sleep-classify plot --fig 6-7`| `outputs/figures/fig6_7_histograms.png` | 主體間個體差異性 Leave-One-Out 交叉驗證直方圖 |
| **Figure 8 & 9**| `uv run sleep-classify plot --fig 8-9`| `outputs/figures/fig8_9_mesa_eval.png` | MESA 獨立外部測試集驗證結果與 ROC 曲線 |

---

## 5. 擴充現代化機器學習模型指南

透過繼承 `BaseSleepClassifier` 抽象策略介面，開發者可以在 5 分鐘內接入全新的機器學習或深度學習模型（如 `XGBoost` 或 `PyTorch 1D-CNN`）。以下是擴充實作範例：

```python
import numpy as np
from xgboost import XGBClassifier
from sleep_classifiers_next.models.base import BaseSleepClassifier

class XGBoostSleepClassifier(BaseSleepClassifier):
    def __init__(self):
        # 封裝 XGBoost 分類器並定義網格搜尋超參數
        clf = XGBClassifier(
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42
        )
        param_grid = {
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.1]
        }
        super().__init__(name="XGBoost", classifier=clf, param_grid=param_grid)

    def train(self, X: np.ndarray, y: np.ndarray, scoring: str = "roc_auc") -> None:
        # 重寫以支援自訂的訓練邏輯或樣本加權
        super().train(X, y, scoring=scoring)
```

---

## 6. 專案架構與目錄樹

```text
sleep_classifiers_next/
├── pyproject.toml                     # Hatchling 建置設定與相依套件宣告
├── README.md                          # 本說明文件
├── data/                              # [Gitignored] 原始感測器與標籤資料
├── outputs/                           # [Gitignored] Parquet 特徵快取與視覺化圖表輸出
│   ├── features/                      # 產生的 {subject_id}_features.parquet
│   └── figures/                       # 生成的論文圖表 PNG 檔
├── sleep_classifiers_next/            # 專案核心原始碼
│   ├── __init__.py
│   ├── cli.py                         # Typer 命令行進入點
│   ├── config.py                      # Pydantic 模組參數驗證與路徑設定
│   ├── etl/                           # 資料抽取、轉換與載入模組
│   │   ├── __init__.py
│   │   ├── activity_count.py          # te Lindert 加速度演算法純 Python 實作
│   │   ├── circadian.py               # Forger 1999 晝夜節律三階 ODE 模擬器
│   │   ├── raw_processor.py           # 原始生理訊號對齊與裁切處理器
│   │   └── feature_pipeline.py        # 逐一主體串流快取與特徵工程流水線
│   ├── evaluation/                    # 驗證與評估模組
│   │   ├── __init__.py
│   │   ├── cross_val.py               # LOSO 與 Monte Carlo 交叉驗證拆分服務
│   │   ├── metrics.py                 # 混淆矩陣、Kappa、AUC、三階段閾值計算
│   │   └── plots.py                   # Matplotlib 論文圖表視覺化復現模組
│   └── models/                        # pluggable 分類器策略模組
│       ├── __init__.py
│       ├── base.py                    # 基礎模型策略類別與 GridSearchCV 整合
│       ├── sklearn_models.py          # Random Forest, MLP, KNN, Logistic Regression
│       └── lgbm_model.py              # LightGBM 分類器策略封裝
└── tests/                             # 自動化單元測試與等價性檢驗
    ├── __init__.py
    ├── test_etl.py                    # ETL 各處理單元之單元測試
    ├── test_models.py                 # 分類器隨插即用訓練與評估測試
    └── test_equivalence.py            # [關鍵] 驗證新舊演算法輸出結果達小數點等價性之測試
```

---

## 7. 致謝與引用

### 論文引用資訊
```bibtex
@article{walch2019sleep,
  title={Sleep stage prediction with raw acceleration and photoplethysmography heart rate data derived from a consumer wearable device},
  author={Walch, Olivia and Huang, Yitong and Forger, Daniel and Sen, Srijan},
  journal={Sleep},
  volume={42},
  number={12},
  pages={zsz180},
  year={2019},
  publisher={Oxford University Press US}
}
```

