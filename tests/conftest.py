import sys
from pathlib import Path

# 將專案根目錄與舊版 sleep_classifiers 加入 sys.path 以防 ModuleNotFoundError
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "sleep_classifiers"))

# 同時在最頂層為 NumPy 2.x 向後相容進行 Monkeypatch
import math
import numpy as np
np.math = math
