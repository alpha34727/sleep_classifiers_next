from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Paths (Default to paths mapped on user's system)
    DATA_DIR: Path = Path("./data")
    OUTPUT_DIR: Path = Path("./outputs")
    
    @property
    def CROPPED_DIR(self) -> Path:
        p = self.OUTPUT_DIR / "cropped"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def FEATURE_DIR(self) -> Path:
        p = self.OUTPUT_DIR / "features"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def FIGURE_DIR(self) -> Path:
        p = self.OUTPUT_DIR / "figures"
        p.mkdir(parents=True, exist_ok=True)
        return p

    # Thresholds & Parameters
    WAKE_THRESHOLD: float = 0.5
    REM_THRESHOLD: float = 0.35
    LOWER_BOUND: float = -0.2
    INCLUDE_CIRCADIAN: bool = False
    REPRODUCE_LEGACY_BUG: bool = True # 這是原論文程式碼的bug，原論文在呼叫 pd.read_csv 時沒有指定 header=None，導致預處理完的資料第一行被當作表頭丟棄。
    REPRODUCE_LEGACY_SPLIT: bool = False # 是否模擬原始專案的 bug，即在 Monte Carlo 分割時不固定隨機 seed 且對不同分類器使用獨立的隨機分割。
    REPRODUCE_LEGACY_GRID_SEARCH_BUG: bool = False # 是否模擬原始專案的 bug，即在 GridSearchCV 內部交叉驗證時，因 sklearn clone 機制導致動態綁定的 class_weight 屬性遺失，使超參數搜尋在不平衡狀態下進行。
    
    model_config = SettingsConfigDict(
        env_prefix="SLEEP_",
        case_sensitive=True,
    )

settings = Settings()

FEATURE_SET_NAMES = {
    "motion": "Motion only",
    "hr": "HR only",
    "motion_hr": "Motion, HR",
    "all": "Motion, HR, and Clock"
}

def normalize_feature_label(feature_set) -> str:
    if isinstance(feature_set, str):
        if feature_set in FEATURE_SET_NAMES:
            return FEATURE_SET_NAMES[feature_set]
            
    s = set(feature_set)
    if s == {"feature_count"}:
        return FEATURE_SET_NAMES["motion"]
    elif s == {"feature_hr"}:
        return FEATURE_SET_NAMES["hr"]
    elif s == {"feature_count", "feature_hr"}:
        return FEATURE_SET_NAMES["motion_hr"]
    elif s == {"feature_count", "feature_hr", "feature_cosine"} or s == {"feature_count", "feature_hr", "feature_circadian"} or s == {"feature_count", "feature_hr", "feature_time"}:
        return FEATURE_SET_NAMES["all"]
    return ", ".join(feature_set)
