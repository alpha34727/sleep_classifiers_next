import numpy as np
from scipy.signal import butter, filtfilt
from sleep_classifiers_next.config import Settings

class ActivityCountProcessor:
    def __init__(self, settings: Settings):
        self.cfg = settings.features

    def compute_activity_counts(self, motion_data: np.ndarray) -> np.ndarray:
        """
        Converts raw triaxial acceleration data to activity counts.
        motion_data: np.ndarray of shape (N, 4) with columns [timestamp, x, y, z]
        Returns: np.ndarray of shape (num_epochs, 2) with columns [timestamp, count]
        """
        fs = self.cfg.motion_fs
        
        # Resample z-axis to regular grid at fs Hz
        t_min, t_max = np.amin(motion_data[:, 0]), np.amax(motion_data[:, 0])
        time_grid = np.arange(t_min, t_max, 1.0 / fs)
        z_interp = np.interp(time_grid, motion_data[:, 0], motion_data[:, 3])

        # Butter bandpass filter
        w1 = self.cfg.motion_butter_passband[0] / (fs / 2)
        w2 = self.cfg.motion_butter_passband[1] / (fs / 2)
        b, a = butter(self.cfg.motion_butter_order, [w1, w2], 'bandpass')
        z_filt = filtfilt(b, a, z_interp)
        z_filt_abs = np.abs(z_filt)

        # Digitization into 128 bins
        bin_edges = np.linspace(
            self.cfg.motion_bin_range[0], 
            self.cfg.motion_bin_range[1], 
            self.cfg.motion_bin_count + 1
        )
        binned = np.digitize(z_filt_abs, bin_edges)

        # Rescale binned values to epochs
        epoch_sec = self.cfg.motion_epoch_seconds
        counts = self._max2epochs(binned, fs, epoch_sec)
        
        # Linear scaling
        counts = (counts - self.cfg.motion_scale_intercept) * self.cfg.motion_scale_slope
        counts[counts < 0] = 0.0

        # Build timestamps for binned counts
        time_counts = np.linspace(t_min, t_max, counts.shape[0])
        return np.column_stack((time_counts, counts))

    def _max2epochs(self, data: np.ndarray, fs: int, epoch: int) -> np.ndarray:
        data = data.flatten()
        seconds = int(np.floor(data.shape[0] / fs))
        data = np.abs(data)
        data = data[0 : seconds * fs]

        # Reshape to (fs, seconds) using column-major order and take max of each second
        data = data.reshape(fs, seconds, order='F').copy()
        data = data.max(0)
        data = data.flatten()

        N = data.shape[0]
        num_epochs = int(np.floor(N / epoch))
        data = data[0 : num_epochs * epoch]

        # Reshape to (epoch, num_epochs) using column-major order and sum each epoch
        data = data.reshape(epoch, num_epochs, order='F').copy()
        epoch_data = np.sum(data, axis=0)
        return epoch_data.flatten()
