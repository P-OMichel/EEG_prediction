import os
import numpy as np
import matplotlib.pyplot as plt
from Functions.time_frequency import spectrogram
from Functions.filter import filter_butterworth



folder_path = 'C:/Users/holcman/Documents/GitHub/EEG_prediction/recordings_FD'  # Change this to your folder path

# List all elements (files and folders)
elements = os.listdir(folder_path)

print(elements)
for elem in elements:

    signal = np.load('recordings_FD/' + elem)
    factor = 10
    signal = signal[::factor]
    fs = int(2500 / factor)
    N = len(signal)
    t = np.linspace(0, N/fs, N)
    #signal = filter_butterworth(signal, fs, [0.1,45])


    print(len(signal))

    t_spectro, f_spectro, spectro = spectrogram(signal, fs, 1, f_cut = fs/2)

    fig, axes = plt.subplots(2, sharex = True)
    axes[0].plot(t, signal)
    axes[1].pcolormesh(t_spectro, f_spectro, np.log(spectro + 0.000001), shading = 'nearest', cmap = 'rainbow', vmin = -3, vmax = 10)

    plt.show()