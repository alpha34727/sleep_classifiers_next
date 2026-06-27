from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Paths (Default to paths mapped on user's system)
    DATA_DIR: Path = Path("C:/Users/alpha/Desktop/sleep_classifiers_next/data")
    OUTPUT_DIR: Path = Path("C:/Users/alpha/Desktop/sleep_classifiers_next/outputs")
    
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
    
    model_config = SettingsConfigDict(
        env_prefix="SLEEP_",
        case_sensitive=True,
    )

settings = Settings()
