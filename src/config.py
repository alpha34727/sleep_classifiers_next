from pathlib import Path

# 專案根目錄
PROJECT_ROOT = Path(__file__).parent.parent

# 數據相關路徑配置
LEGACY_CROPPED_DIR = PROJECT_ROOT / "sleep_classifiers" / "outputs" / "cropped"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# 特徵欄位定義
FEATURE_COLS = ["count_feature", "hr_feature", "time_feature", "cosine_feature"]
TARGET_COL = "stage"

# 睡眠標籤轉換配置
# 原始標記：0=Wake, 1=N1, 2=N2, 3=N3, 4=N4, 5=REM, -1=Unscored
# 二分類 (Sleep/Wake)：Wake -> 0, Sleep -> 1
BINARY_LABEL_MAP = {
    0: 0,  # Wake
    1: 1,  # N1 -> Sleep
    2: 1,  # N2 -> Sleep
    3: 1,  # N3 -> Sleep
    4: 1,  # N4 -> Sleep
    5: 1,  # REM -> Sleep
}

# 三分類 (Wake/NREM/REM)：Wake -> 0, NREM -> 1, REM -> 2
THREE_CLASS_LABEL_MAP = {
    0: 0,  # Wake
    1: 1,  # N1 -> NREM
    2: 1,  # N2 -> NREM
    3: 1,  # N3 -> NREM
    4: 1,  # N4 -> NREM
    5: 2,  # REM
}
