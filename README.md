# 🌙 sleep_classifiers-next

> **Sleep Stage Prediction Using Wearable Data - Modern Refactored Version**  
> 基於 Apple Watch 運動與心率數據的睡眠分期預測演算法（全新現代化重構與演算法升級版）

---

## 📖 專案簡介

`sleep_classifiers-next` 是一個將原著 Olivia Walch 團隊開發的 Apple Watch 睡眠分期預測演算法（發表於學術論文 *[zsz180.pdf](file:///c:/Users/Johnsou/Desktop/sleep_classifiers_next/sleep_classifiers/zsz180.pdf)*）利用現代軟體工程方法進行全面重構與演算法升級的專案。

本專案保留了原著的核心訊號處理與物理建模邏輯（如差分高斯 DoG 濾波器、雙向高斯平滑卷積、餘弦時間特徵等），並透過現代化 Python 工具鏈進行重寫，解決了舊版專案記憶體吞吐量極大、無法並行計算、磁碟 I/O 效率差等痛點。同時，本專案全新整合了現代梯度提升樹（GBDT）模型，並提供了物件導向的交叉驗證與高品質學術圖表繪製流程。

---

## 🚀 重構亮點與效能革命

本專案對舊版程式碼進行了深度的軟體工程重構，其核心效能與功能提升對比如下：

| 對比維度 | 舊版專案 (Legacy) | 新版專案 (Next-Gen) | 重構技術與效益 |
| :--- | :--- | :--- | :--- |
| **記憶體佔用 (RAM)** | **150GB+** (需一次性載入所有膨脹的中間特徵) | **< 1GB** (受試者級別離線特徵提取與 Polars 惰性求值) | 降低 99% 以上記憶體需求，一般開發設備即可流暢執行。 |
| **運行環境 (Env)** | Python 3.7 + 舊版 scikit-learn | **Python 3.12+** | 使用現代化 **uv / poetry** 進行嚴格且高效率的依賴管理。 |
| **資料儲存格式** | CSV / Pickle 序列化 | **Parquet** 格式 | 具備高壓縮比與高快取效率，大幅降低硬碟 I/O 開銷與讀寫時間。 |
| **支援演算法 (Models)** | 僅支援 Logistic Regression, Random Forest, KNN, MLP | 完美相容經典模型，並**整合現代主流 GBDT（XGBoost, LightGBM, CatBoost）** | 提供統一的 `BaseSleepClassifier` 抽象介面，便於快速新增或更換分類器。 |
| **計算與評估效率** | 單執行緒運算，存在嚴重 CPU 與 I/O 瓶頸 | **多核平行處理（Parallelized Monte Carlo CV）** | 採用 `joblib` 平行化引擎，可完美吃滿所有 CPU 核心，計算速度提升數倍。 |
| **程式碼架構** | 腳本式、程式碼冗餘度高 | **物件導向設計 (OO-Design)** | 結構清晰，將特徵建構、模型封裝、評估驗證與圖表繪製完全解耦。 |

---

## 📁 專案目錄結構

本專案遵循清晰、規範的機器學習專案目錄結構：

```text
sleep_classifiers_next/
├── data/
│   ├── raw/                 # 原始數據（需從 PhysioNet 下載，包含 heart_rate, labels, motion, steps）
│   ├── interim/             # 中間過渡數據（可選）
│   └── processed/           # 經特徵工程處理後的受試者 Parquet 數據
├── outputs/
│   └── figures/             # 輸出的論文對齊圖表（ROC, PR 曲線等）
├── src/
│   ├── __init__.py
│   ├── config.py            # 全域路徑及標籤映射配置
│   ├── data_loader.py       # 高效數據載入器 (支援記憶體快取與多分類/二分類標籤轉換)
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── plotter.py       # 繪圖工具（ROC, PR 曲線、Bland-Altman 圖、混淆矩陣）
│   │   └── validator.py     # 交叉驗證引擎（LOOCV 與 Monte Carlo 交叉驗證，支援並行計算）
│   ├── features/
│   │   ├── __init__.py
│   │   ├── builder.py       # 離線特徵建構管線（Polars/NumPy 向量化加速）
│   │   └── signal_processing.py # 訊號處理核心（DoG 帶通濾波器、雙向高斯平滑卷積）
│   └── models/
│       ├── __init__.py
│       ├── base.py          # 睡眠分類器統一抽象基底類別
│       ├── baseline.py      # 經典機器學習模型 (Logistic Regression, Random Forest, MLP, k-NN)
│       └── modern_gbdt.py   # 現代梯度提升模型 (LightGBM, XGBoost)
├── tests/
│   ├── test_alignment.py   # 新舊特徵等價性/對齊驗證測試
│   ├── test_models.py      # 模型訓練與 API 介面測試
│   └── test_validation.py  # 交叉驗證與指標計算測試
├── figures_mc_sleep_wake.py # 執行多模型/多特徵交叉驗證並繪製 Figure 2/3 的主程式
├── run_preprocessing.py     # 特徵前處理一鍵執行腳本
├── main.py                  # 全模型交叉驗證與指標輸出主程式
├── pyproject.toml           # 現代 Python 專案依賴管理配置
└── README.md                # 本說明文件
```

---

## 📦 環境安裝指引

本專案推薦使用現代化 Python 套件管理器 [uv](https://github.com/astral-sh/uv) 進行快速環境配置。

### 1. 複製本專案
```bash
git clone https://github.com/your-username/sleep_classifiers-next.git
cd sleep_classifiers-next
```

### 2. 建立並啟用虛擬環境
使用 `uv` 建立虛擬環境：
```bash
# Windows
uv venv
.venv\Scripts\activate

# macOS / Linux
uv venv
source .venv/bin/activate
```

### 3. 以可編輯模式安裝專案及依賴
```bash
uv pip install -e .
# 或使用標準 pip 安裝：
pip install -e .
```

---

## 🏃‍♂️ 執行流水線

在執行流水線之前，請參閱 [data/README.md](file:///c:/Users/Johnsou/Desktop/sleep_classifiers_next/data/README.md) 下載原始 PhysioNet 睡眠數據並放置在指定位置。

### 步驟一：特徵工程（離線提取）
本步驟將讀取原始運動與心率數據，透過 Polars 高效計算 DoG 濾波與時間特徵，並為每位受試者單獨輸出一個高壓縮率的 Parquet 檔案。
```bash
python run_preprocessing.py
```
*提示：預設會自動跳過已處理過的受試者。若要強制重新計算，請加入 `--no-skip` 參數。*

### 步驟二：執行全模型評估與訓練
本步驟會對所有分類器模型（Logistic Regression, Random Forest, MLP, k-NN, LightGBM, XGBoost）執行 Monte Carlo 交叉驗證（MCCV），並在終端機輸出 Markdown 格式的完整指標對比表格。
```bash
python main.py --n-splits 20 --n-jobs -1
```

### 步驟三：生成論文對齊圖表
本步驟將在 `outputs/figures/` 目錄下繪製並生成與原論文對齊的 **Figure 2 (ROC 網格圖)** 與 **Figure 3 (Precision-Recall 網格圖)**。
```bash
python figures_mc_sleep_wake.py --n-splits 20 --n-jobs -1
```

---

## 🧪 自動化測試與等價性驗證

為了確保重構後的 NumPy / Polars 向量化特徵工程與原著論文及舊版專案在數學計算上完全一致，本專案提供了一套完整的等價性測試套件。

執行特徵對齊驗證（DoG 帶通濾波器、雙向高斯卷積等）：
```bash
pytest tests/test_alignment.py
```
*測試中設定了嚴格的容差 `atol=1e-5` 與 `rtol=1e-5`，以確保特徵計算結果在浮點數精度範圍內完美對齊。*

執行所有單元測試與 API 測試：
```bash
pytest
```

---

## 📊 免責聲明與致謝

### 致謝 (Acknowledgements)
感謝原作者 **Olivia Walch** 團隊為學術界提供的 Apple Watch 睡眠分類研究與開源程式碼貢獻。

### 引用文獻 (Citations)
若您在學術研究中使用了此預測模型或其衍生版本，請引用原著論文：
```text
Walch, O., Huang, Y., Forger, D. B., & Goldstein, C. (2019). 
Sleep stage prediction with raw acceleration and heart rate data from a consumer wearable device. 
Sleep, 42(12), zsz180.
```
