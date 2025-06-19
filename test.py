import numpy as np
import scipy as sc
import pandas as pd
import matplotlib.pyplot as plt
from Functions.entropy import *
from Functions.dimension import *
from Functions.open_file import open_clean_records
from Functions.time_frequency import spectrogram

#--- load data
name = 'recordings/rec_20240201_110448.mat'
fs = 128
signal, t = open_clean_records(name,fs,True,True) 

# i = 2055 * fs
# j = 2057 * fs
# signal = np.concatenate((signal[:i], signal[j:]))
# t = np.concatenate((t[:i], t[j:]))

#--- Compute spectrogram
t_spectro, f_spectro, spectro = spectrogram(signal, fs, 0.25)

#--- Sliding window setup
window_size = int(fs * 30)
step_size = window_size // 6
n_windows = (len(signal) - window_size) // step_size + 1

metrics_evolution = {
    "Shannon (Fixed Bins)": [],
    "Shannon (KDE)": [],
    "Differential Entropy (KDE)": [],
    "Approximate Entropy": [],
    "Sample Entropy": [],
    "Permutation Entropy": [],
    "Spectral Entropy": [],
    "Correlation Dimension": [],
    "Hurst exponent": []
}
window_centers = []

for i in range(n_windows):
    start = i * step_size
    end = start + window_size
    window = signal[start:end]
    window_centers.append(start + window_size // 2)

    # metrics_evolution["Shannon (Fixed Bins)"].append(shannon_entropy_fixed_bins(window))
    # metrics_evolution["Shannon (KDE)"].append(shannon_entropy_kde(window))
    # metrics_evolution["Differential Entropy (KDE)"].append(differential_entropy_kde(window))
    # metrics_evolution["Approximate Entropy"].append(compute_approximate_entropy(window))
    metrics_evolution["Sample Entropy"].append(compute_sample_entropy(window, 2))
    # metrics_evolution["Permutation Entropy"].append(compute_permutation_entropy(window))
    # metrics_evolution["Spectral Entropy"].append(compute_spectral_entropy(window))
    metrics_evolution['Correlation Dimension'].append(correlation_dimension(window))
    metrics_evolution['Hurst exponent'].append(hurst_exponent(window))

# # Plot
# plt.figure(figsize=(12, 8))
# for key, values in metrics_evolution.items():
#     plt.plot(window_centers, values, label=key)

# plt.xlabel("Time (center index)")
# plt.ylabel("Metrics")
# plt.title("Sliding-Window Entropy Evolution")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

#--- Display
fig, axes = plt.subplots(5, sharex = True)
axes[0].plot(t, signal)
axes[0].set_ylim(-75, 75)
axes[1].pcolormesh(t_spectro, f_spectro, np.log(spectro + 0.00001), shading = 'nearest', cmap = 'rainbow', vmin = np.log(0.001), vmax = np.log(20))
axes[2].plot(t[window_centers], metrics_evolution["Sample Entropy"], label = 'Sample Entropy')
axes[2].plot(t[window_centers], sc.signal.savgol_filter(metrics_evolution["Sample Entropy"], 30, 1))
axes[2].legend()
axes[3].plot(t[window_centers], metrics_evolution['Correlation Dimension'], label = 'Correlation Dimension')
axes[3].plot(t[window_centers], sc.signal.savgol_filter(metrics_evolution["Correlation Dimension"], 30, 1))
axes[3].legend()
axes[4].plot(t[window_centers], metrics_evolution['Hurst exponent'], label = 'Hurst exponent')
axes[4].plot(t[window_centers], sc.signal.savgol_filter(metrics_evolution["Hurst exponent"], 30, 1))
axes[4].legend()

plt.show()