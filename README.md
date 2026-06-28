# Sleep Classifiers Next: 現代化穿戴式裝置睡眠階段預測系統

本專案是針對牛津大學睡眠研究學會（Oxford Sleep Research Society）於 2019 年發表的經典學術論文進行的現代化、生產級重構：
> **論文引用**：*Walch et al., "Sleep stage prediction with raw acceleration and photoplethysmography heart rate data derived from a consumer wearable device", SLEEPJ 2019.*

本系統旨在利用消費級智慧穿戴裝置所獲取的原始三軸加速度（Raw Acceleration）與光電容積脈搏波（PPG）心率資料，進行高精度的睡眠階段預估（包含二分類：睡眠/清醒，以及三分類：清醒/NREM/REM）。

---

## 核心突破與現代化亮點

本專案（`sleep_classifiers_next`）相較於 2019 年的舊版程式碼，實現了以下四大工程突破：

1. **極致記憶體最佳化 (Memory Win)**
   * **痛點**：舊版程式碼在處理長時段滑動視窗（Windowing）時，會產生大量的資料複製與冗餘物件，導致記憶體消耗成指數級爆炸，處理大規模資料集時需要高達 `150GB+` 的記憶體。
   * **解決方案**：全面導入 **Polars Lazy Evaluation (惰性求值)** 機制，透過查詢優化器（Query Optimizer）最小化記憶體分配與複製，並搭配記憶體對應（Memory Mapping）與高效分塊（Chunking）串流。
   * **成效**：在維持完全相同的特徵提取品質下，將峰值記憶體消耗嚴格控制在 **`2GB` 以下**。

2. **高速資料 ETL 管線**
   * 基於現代 **Polars** 與 **PyArrow Parquet 串流** 重新設計資料讀取與轉換管線。
   * 淘汰舊版已過時的 Pandas 0.24 框架，資料讀寫效率提升數十倍，同時獲得更好的型別完整性與平行處理能力。

3. **策略模式模型架構 (Strategy Pattern)**
   * 精心設計 `BaseClassifierStrategy` 抽象介面，解耦模型定義、訓練與推論邏輯。
   * **相容性與可擴充性**：完美封裝並支援學術復現所需的舊版 Scikit-Learn 傳統模型（隨機森林、邏輯斯迴歸、k-近鄰、多層感知機），並可一鍵抽換至現代高速梯度提升樹模型（**LightGBM** 與 **XGBoost**）。

4. **可解釋性引進 (SHAP Interpretability)**
   * 整合 **TreeSHAP** 演算法，針對現代提升樹模型（如 LightGBM）進行細粒度的特徵貢獻度解碼。
   * 能夠量化評估活動量計數（Motion）、心率變異性（HRV）與晝夜節律代理變數（Circadian Clock Proxy）對各睡眠階段分類決策的邊際貢獻。

---

## 新舊架構深度對比 (Architectural Comparison)

### 1. 目錄結構演進

我們將舊版散落的指令碼重構為結構清晰的 Python 套件 `sleep_next`，並將可執行指令碼獨立置於 `scripts` 目錄中：

```text
Legacy Repo (2019)                        Modernized Repo (sleep_classifiers_next)
======================================    ======================================
sleep_classifiers/                       sleep_classifiers_next/
├── Requirements.txt                      ├── pyproject.toml
├── requirements.txt                      ├── uv.lock
├── preprocessing_runner.py               ├── scripts/
├── analysis_runner.py                    │   ├── 01_run_preprocessing.py
├── program_flow.md                       │   ├── 02_run_analysis.py
├── source/                               │   └── 03_reproduce_all.py
│   ├── constants.py                      ├── src/
│   ├── utils.py                          │   └── sleep_next/
│   ├── sleep_stage.py                    │   │   ├── __init__.py
│   ├── make_counts.m                     │   │   ├── config.py
│   ├── preprocessing/                    │   │   ├── data/
│   │   ├── raw_data_processor.py         │   │   │   ├── loader.py
│   │   ├── time_service.py               │   │   │   └── preprocess.py
│   │   ├── epoch.py                      │   │   ├── evaluate/
│   │   ├── interval.py                   │   │   │   ├── cross_val.py
│   │   │   ... (多個子目錄)                │   │   │   └── metrics.py
│   └── analysis/                         │   │   ├── features/
│       ... (多個分析程式)                  │   │   │   ├── clock_proxy.py
│                                         │   │   │   └── motion.py
│                                         │   │   ├── models/
│                                         │   │   │   ├── base.py
│                                         │   │   │   ├── legacy.py
│                                         │   │   │   └── modern.py
│                                         │   │   └── visualize/
│                                         │   │       └── plotters.py
│                                         └── tests/
```

### 2. 架構特質對比

| 比較維度 | Legacy Repo (2019) | Modernized Repo (`sleep_classifiers_next`) | 演進效益 |
| :--- | :--- | :--- | :--- |
| **環境管理** | Python 3.7 + `requirements.txt` | Python 3.12 + `uv` | 極速解析、嚴格鎖定相依版本、工作流完全重現 |
| **資料引擎** | Pandas (Eager Load) 全記憶體載入 | Polars (LazyFrame + Chunking) 惰性串流 | 記憶體需求自 150GB+ 急劇下降至 **<2GB** |
| **模型封裝** | 硬編碼（Hardcoded）散落的訓練與推論邏輯 | `BaseClassifierStrategy` 統一抽象類別 | 實踐策略模式，可一鍵抽換 XGBoost / LightGBM |
| **多階分類** | 寫死於分析腳本中，缺乏彈性 | 動態二分搜尋相等準確率引擎 (Equal Accuracy) | 嚴格復現論文 Table 6 的動態自適應閾值調整 |
| **型別安全** | 無（原生 Python 弱型別與動態屬性） | Pydantic Settings + Type Hints 強型別 | 杜絕靜默配置錯誤與執行期型別推導異常 |

---

## 學術復現度保證 (Mathematical Parity Guarantee)

雖然我們全面重構了底層的 ETL 與物件架構，但針對生醫訊號處理的數學算子與視窗切片邏輯，本專案保證與原始論文具有 **1:1 的嚴格數學等價性**：

* **高斯差分濾波器 (Difference of Gaussians, DoG)**
  心率訊號預處理使用雙高斯卷積核（標準差 $\sigma_1 = 120 \text{s}$，$\sigma_2 = 600 \text{s}$，縮放因子 $0.75$）進行帶通濾波，確保心率變異特徵與論文完全一致。
* **活動量計數高斯平滑**
  利用高斯核（標準差 $\sigma = 50 \text{s}$）進行活動計數（Activity Count）卷積平滑。
* **時間視窗切片 (Window Slicing)**
  以當前 Epoch 為中心，採用 $\pm 5 \text{min}$（即 $10 \times 30 \text{s} - 15 \text{s} = 285 \text{s}$ 前後擴展，共計 $600 \text{s}$ 或 10 分鐘寬度）的滑動視窗進行局部特徵描述統計。


---

## 圖表產生指南


### 第一步：下載專案原始碼 (Clone Repository)
1. 在 Windows 系統中，按下鍵盤上的 **`Win + X`** 鍵，在選單中選擇 **「終端機」** 或 **「PowerShell」**。
2. 確保您已經安裝了 Git。執行以下指令將專案下載到您的電腦中（此處以桌面為例）：
   ```bash
   cd ~\Desktop
   git clone https://github.com/alpha34727/sleep_classifiers_next
   ```
*(註：請將網址替換為實際的專案 Git 網址)*

### 第二步：安裝 `uv`（環境管理工具）
1. 在同一個終端機視窗中，複製以下指令並貼上，然後按下 `Enter` 鍵執行：
   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```
2. 指令執行完畢後，**請關閉該終端機視窗，並重新開啟一個新的終端機視窗**（這樣新安裝的 `uv` 工具才能生效）。

### 第三步：取得與設定資料集 (Dataset Setup)
本專案使用 PhysioNet 開源的 Sleep Accel 資料集：
1. 點擊連結前往下載頁面：[PhysioNet Sleep Accel Dataset 1.0.0](https://physionet.org/content/sleep-accel/1.0.0/)。
2. 下載資料壓縮檔（可直接下載 `sleep-accel-raw-acceleration-and-photoplethysmography-heart-rate-data-derived-from-a-consumer-wearable-device-1.0.0.zip`）。
3. 下載後將其解壓縮。
4. 進入剛剛下載的專案資料夾 `sleep_classifiers_next`，找到名為 `data` 的資料夾。
5. 將解壓出來的 **`heart_rate`**、**`labels`**、**`motion`**、**`steps`** 這四個資料夾，複製並貼至本專案的 **`data`** 資料夾下。

完成後，請確認您的 `data` 資料夾內部結構如下所示（必須包含這些子資料夾與文字檔案）：
```text
sleep_classifiers_next/
└── data/
    ├── heart_rate/
    │   ├── 46343_heartrate.txt
    │   └── ...
    ├── labels/
    │   ├── 46343_labeled_sleep.txt
    │   └── ...
    ├── motion/
    │   ├── 46343_acceleration.txt
    │   └── ...
    └── steps/
        ├── 46343_steps.txt
        └── ...
```

### 第四步：切換路徑並自動建置環境
1. 在新的終端機中，使用 `cd` 指令切換至本專案所在的資料夾。例如：
   ```powershell
   cd C:\Users\alpha\Desktop\sleep_classifiers_next
   ```
2. 執行以下指令。`uv` 將會自動偵測設定，下載並安裝對應的 Python 版本與本專案所需的所有軟體套件：
   ```bash
   uv sync
   ```

### 第五步：執行資料預處理 (ETL)
1. 執行以下指令，系統會將原始的 `.txt` 檔案進行資料對齊、剪裁、高斯平滑濾波處理，並轉換為高效能的 Parquet 格式：
   ```bash
   uv run scripts/01_run_preprocessing.py
   ```
2. 畫面會顯示處理進度。預處理完成後，您會在專案中看到新產生的 `outputs/cropped/` 與 `outputs/features/` 資料夾，裡面存放著處理好的特徵檔案。

### 第六步：執行分析並自動產生所有圖表
執行以下指令，系統將會使用 6 種機器學習模型（包含隨機森林、XGBoost、LightGBM 等）進行蒙地卡羅交叉驗證，並在完成後自動產出學術圖表與對比報告：
```bash
uv run scripts/03_reproduce_all.py --binary-splits 50 --three-class-splits 20
```
*(提示：此標竿測試計算量較大，大約需要 5-15 分鐘，請耐心等待其執行完畢。)*

---

## 產出圖表與報告位置指引

當上述步驟執行完畢後，所有產出的學術圖表與效能數據報告都會自動儲存在專案根目錄下的 **`outputs`** 資料夾中：

### 1. 學術分析圖表 (`outputs/figures/` 目錄)
您可以在此資料夾下找到與論文對等的關鍵圖表：
* **`combined_sw_roc.png`**：所有模型在二分類（睡眠/清醒）任務下的 ROC 接收者操作特徵曲線網格圖。
* **`combined_sw_pr.png`**：所有模型在二分類（睡眠/清醒）任務下的 Precision-Recall 準確率-召回率曲線網格圖。
* **`modern_lightgbm_shap_summary.png`**：LightGBM 模型的 TreeSHAP 特徵貢獻度解釋圖，直觀展示 Motion、HR 與時間特徵對預測的影響力。
* **`[ClassifierName]_three_class_roc.png`**：指定模型在三分類（清醒/NREM/REM）任務下，尋找最佳平衡準確率閾值的曲線圖。
* **`[ClassifierName]_bland_altman.png`**：預估總睡眠時間（TST）與 PSG 黃金標準的 Bland-Altman 偏差一致性分析散點圖。

### 2. 標竿對比報告 (`outputs/tables/` 目錄)
* **`full_paper_benchmark.md`**：一份自動生成的 Markdown 報告。其中包含與論文 Table 2 至 Table 6 完全對等的詳細性能對比表格，詳細記錄了不同模型在各個特徵子集上的 Accuracy、Specificity (Wake Correct)、Sensitivity (Sleep Correct)、Cohen's Kappa 以及 AUC 數值。

---

## 致謝 (Acknowledgments)

本專案的工程重構之旅，完全建立在原著團隊遠大的學術願景與無私的開源精神之上。在此誠摯致謝：

* **原著論文研究團隊**：**Dr. Olivia Walch, Yitong Huang, Daniel Forger 與 Dr. Cathy Goldstein**。在消費級穿戴式裝置演算法普遍被視為商業機密、原始感測器數據極度封閉的 2019 年，團隊展現了令人敬佩的學術勇氣，率先開源了透過 Apple Watch 原始加速度計與 PPG 心率推估睡眠階段的完整數學公式。
* **資料集基礎設施**：感謝美國密西根大學睡眠與時間生理學實驗室（University of Michigan Sleep and Chronophysiology Laboratory）以及 **PhysioNet** 平台，為本研究提供了極其寶貴、且伴隨 PSG 黃金標準交叉驗證的開放資料集。

*「如果我們能看得更遠，那是因為我們站在巨人的肩膀上。」* —— 謹以 `sleep_classifiers_next` 現代化專案，向原作者團隊致以最深的敬意。

---

## 授權與引用須知 (License & Citation)

### 雙層授權聲明 (Dual-Layer Licensing)
本專案採用雙層授權模式，確保學術貢獻的歸屬並賦予軟體工程架構最大的開源自由度：
1. **軟體工程架構 (MIT License)**：本專案引入之現代化程式架構、ETL 資料管線、策略模式與封裝設計，皆採用寬鬆的 [MIT 授權條款](LICENSE)。
2. **學術演算法本體 (CC BY-NC 4.0)**：底層的睡眠階段分類方法論、高斯差分數學濾波器、模型標竿參數等核心知識產權，仍受原始論文之 **創用 CC 姓名標示-非商業性 4.0 國際授權條款 (CC BY-NC 4.0)** 規範。

> ⚠️ **嚴禁商業用途紅線**：若第三方開發者或企業欲將本專案之預測模型或其衍生應用，整合至任何商業化穿戴式裝置、付費應用程式或其他具營利性質之產品服務中，必須事先取得**牛津大學出版社 (Oxford University Press)** 的獨立書面授權（聯繫信箱：`journals.permissions@oup.com`）。

### 學術引用格式 (BibTeX)
如果您在研究中使用了本專案的程式碼或演算法架構，請務必引用原始論文，以對原作者的研究貢獻表達最崇高的敬意：
```bibtex
Olivia Walch, Yitong Huang, Daniel Forger, Cathy Goldstein, Sleep stage prediction with raw acceleration and photoplethysmography heart rate data derived from a consumer wearable device, Sleep, Volume 42, Issue 12, December 2019, zsz180, https://doi.org/10.1093/sleep/zsz180
```
