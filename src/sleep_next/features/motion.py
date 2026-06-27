import numpy as np
from scipy.signal import butter, filtfilt

def max2epochs(data: np.ndarray, fs: int, epoch_length: int) -> np.ndarray:
    data = np.abs(data.flatten())
    seconds = int(np.floor(len(data) / fs))
    data = data[:seconds * fs]
    
    # Reshape in Fortran order to group by seconds: shape (fs, seconds)
    data = data.reshape(fs, seconds, order='F')
    data = data.max(axis=0) # max of each second
    
    N = len(data)
    num_epochs = int(np.floor(N / epoch_length))
    data = data[:num_epochs * epoch_length]
    
    # Reshape in Fortran order to group by epoch: shape (epoch_length, num_epochs)
    data = data.reshape(epoch_length, num_epochs, order='F')
    epoch_data = data.sum(axis=0)
    
    return epoch_data.astype(np.float32)

def compute_activity_counts(timestamps: np.ndarray, z_acc: np.ndarray) -> np.ndarray:
    """
    Computes activity counts from raw accelerometer z-axis data.
    Aligns with ActivityCountService.build_activity_counts_without_matlab.
    """
    if len(timestamps) == 0:
        return np.empty((0, 2), dtype=np.float32)
        
    fs = 50
    t_min = np.amin(timestamps)
    t_max = np.amax(timestamps)
    
    # Interpolate to 50 Hz
    time_50hz = np.arange(t_min, t_max, 1.0 / fs)
    z_data = np.interp(time_50hz, timestamps, z_acc)
    
    # Butter passband filter (3Hz to 11Hz at 50Hz sample rate)
    cf_low = 3.0
    cf_hi = 11.0
    order = 5
    w1 = cf_low / (fs / 2.0)
    w2 = cf_hi / (fs / 2.0)
    b, a = butter(order, [w1, w2], btype='bandpass')
    
    z_filt = filtfilt(b, a, z_data)
    z_filt = np.abs(z_filt)
    
    # Binning logic
    top_edge = 5.0
    bottom_edge = 0.0
    number_of_bins = 128
    bin_edges = np.linspace(bottom_edge, top_edge, number_of_bins + 1)
    binned = np.digitize(z_filt, bin_edges)
    
    # Max to epochs (15-second epoch counts)
    epoch_len = 15
    counts = max2epochs(binned, fs, epoch_len)
    
    # Scaling counts
    counts = (counts - 18.0) * 3.07
    counts[counts < 0.0] = 0.0
    
    # Linspace epoch timestamps
    time_counts = np.linspace(t_min, t_max, len(counts))
    
    output = np.column_stack((time_counts.astype(np.float32), counts.astype(np.float32)))
    return output
