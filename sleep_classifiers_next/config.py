import os
from pathlib import Path
from pydantic import BaseModel, Field

class PathSettings(BaseModel):
    data_dir: Path = Field(default=Path("c:/Users/alpha/Desktop/sleep_classifiers_next/data"))
    output_dir: Path = Field(default=Path("c:/Users/alpha/Desktop/sleep_classifiers_next/outputs"))

    @property
    def cropped_dir(self) -> Path:
        return self.output_dir / "cropped"

    @property
    def features_dir(self) -> Path:
        return self.output_dir / "features"

    @property
    def figures_dir(self) -> Path:
        return self.output_dir / "figures"

    def make_dirs(self) -> None:
        self.cropped_dir.mkdir(parents=True, exist_ok=True)
        self.features_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

class FeatureSettings(BaseModel):
    epoch_duration: int = 30
    seconds_per_minute: int = 60
    seconds_per_hour: int = 3600
    seconds_per_day: int = 86400
    
    # Activity counts settings
    motion_fs: int = 50
    motion_butter_order: int = 5
    motion_butter_passband: list[float] = [3.0, 11.0]
    motion_bin_range: list[float] = [0.0, 5.0]
    motion_bin_count: int = 128
    motion_epoch_seconds: int = 15
    motion_scale_slope: float = 3.07
    motion_scale_intercept: float = 18.0
    motion_gaussian_sigma: float = 50.0  # seconds
    motion_window_size: int = 10 * 30 - 15  # 285s on either side

    # Heart rate settings
    hr_window_size: int = 10 * 30 - 15  # 285s
    hr_dog_sigma1: float = 120.0  # seconds
    hr_dog_sigma2: float = 600.0  # seconds
    hr_dog_scalar: float = 0.75

    # Circadian proxy settings
    circadian_tau: float = 24.2
    circadian_G: float = 19.875
    circadian_k: float = 0.55
    circadian_mu: float = 0.23
    circadian_b: float = 0.013
    circadian_a0: float = 0.16
    circadian_I0: float = 9500.0
    circadian_p: float = 0.6
    circadian_steps_threshold: int = 20
    circadian_days_to_entrain: int = 60
    circadian_rk4_dt: float = 0.1  # hours
    circadian_lower_bound: float = -0.2
    legacy_compatibility: bool = False

class ModelSettings(BaseModel):
    wake_threshold: float = 0.5
    rem_threshold: float = 0.35
    random_seed: int = 42

class Settings(BaseModel):
    paths: PathSettings = Field(default_factory=PathSettings)
    features: FeatureSettings = Field(default_factory=FeatureSettings)
    models: ModelSettings = Field(default_factory=ModelSettings)
    verbose: bool = True
