import numpy as np
import matplotlib.pyplot as plt
from Functions.entropy import *
from Functions.open_file import open_clean_records
from Functions.time_frequency import spectrogram
from Functions.filter import filter_butterworth
from Functions.utils import sobolev_distance

#--- load data
name = 'recordings/rec_20240321_085300.mat'
fs = 128
signal, t = open_clean_records(name,fs,True,True) 
signal = filter_butterworth(signal, fs, [0.1,45])

#--- Compute spectrogram
t_spectro, f_spectro, spectro = spectrogram(signal, fs, 0.25)

# for rec_20240321_085300 | [500,700,900,1300,1500,1800,2100,2400]

start_lists = [500,700,900,1300,1500,1800,2050,2400]
ws = 60
windows = [signal[start*fs:(start+ws)*fs] for start in start_lists]
print(len(windows[0]))

scale_max = 5

N_windows = len(windows)
fig, axes = plt.subplots(N_windows+1)
axes[0].pcolormesh(t_spectro, f_spectro, np.log(spectro + 0.00001), shading = 'nearest', cmap = 'rainbow', vmin = np.log(0.001), vmax = np.log(20))
for start in start_lists:
    axes[0].axvline(start, color = 'black')

for i in range(N_windows):
    
    y = windows[i]
    #--- compute multiscale entropy metrics
    mse = multiscale_entropy(y, max_scale=scale_max)
    mpe = multiscale_permutation_entropy(y, max_scale=scale_max)

    # compute integral of cumulative
    int_cumulative_mse = round(get_int_cumulative(mse), 2)
    int_cumulative_mpe = round(get_int_cumulative(mpe), 2)

    # compute integral of cumulative of difference
    int_cumulative_diff = round(get_int_cumulative_diff(mse, mpe), 2)

    # analysis:
    analysis = analyze_curves(mse, mpe)
    print('current window: ', i)
    print(analysis)

    #--- plots
    axes[i+1].plot(range(1, scale_max + 1), mse, label=" MSE | " + str(int_cumulative_mse), marker='o')
    axes[i+1].plot(range(1, scale_max + 1), mpe, label=" MPE | " + str(int_cumulative_mpe), marker='o')
    axes[i+1].plot(0,0,label=" diff | " + str(int_cumulative_diff), marker='o')
    #axes[i+1].set_title(sobolev_distance(np.array(mse),np.array(mpe)))
    axes[i+1].legend()

fig.tight_layout()

plt.show()