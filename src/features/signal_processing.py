import numpy as np

def smooth_gauss(y: np.ndarray, box_pts: int) -> float:
    """計算給定視窗 y 的 Gaussian 加權平均值，完全對齊舊版 utils.smooth_gauss。"""
    box = np.ones(box_pts) / box_pts
    mu = int(box_pts / 2.0)
    sigma = 50.0  # seconds

    for ind in range(0, box_pts):
        box[ind] = np.exp(-0.5 * (((ind - mu) / sigma) ** 2))

    box = box / np.sum(box)
    return float(np.sum(box * y))

def convolve_with_dog(y: np.ndarray, box_pts: int) -> np.ndarray:
    """心率序列的差分高斯卷積，完全對齊舊版 utils.convolve_with_dog，包括其特殊的邊界 Padding bug。"""
    y = y - np.mean(y)
    box = np.ones(box_pts) / box_pts

    mu1 = int(box_pts / 2.0)
    sigma1 = 120

    mu2 = int(box_pts / 2.0)
    sigma2 = 600

    scalar = 0.75

    for ind in range(0, box_pts):
        box[ind] = np.exp(-0.5 * (((ind - mu1) / sigma1) ** 2)) - scalar * np.exp(
            -0.5 * (((ind - mu2) / sigma2) ** 2)
        )

    # 複製舊版特殊的 np.insert padding 行為
    # 1. 頭部 padding (142 個元素)
    pad_start = np.flip(y[0:int(box_pts / 2)])
    y = np.insert(y, 0, pad_start)

    # 2. 尾部 padding (142 個元素)
    # 注意，y 的長度在此時已經變大，使用 len(y) - 1 會把 pad_end 插入在倒數第二個元素之前，
    # 這是舊版原始代碼的精確實現。
    pad_end = np.flip(y[int(-box_pts / 2):])
    y = np.insert(y, len(y) - 1, pad_end)

    y_smooth = np.convolve(y, box, mode='valid')
    return y_smooth
