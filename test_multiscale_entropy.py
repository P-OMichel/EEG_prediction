import numpy as np
import matplotlib.pyplot as plt
from Functions.entropy import multiscale_entropy, refined_composite_multiscale_entropy, multiscale_permutation_entropy
from Functions.open_file import open_clean_records
from Functions.time_frequency import spectrogram

#--- load data
name = 'recordings/rec_20240307_115040.mat'
fs = 128
signal, t = open_clean_records(name,fs,True,True) 

#--- Compute spectrogram
t_spectro, f_spectro, spectro = spectrogram(signal, fs, 0.25)

# for rec_20240321_085300 | [500,700,900,1300,1500,1800,2100,2400]

start_lists = [500,700,900,1300,1500,1800,2100,2400]
windows = [signal[start*fs:(start+100)*fs] for start in start_lists]

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
    rcmse = refined_composite_multiscale_entropy(y, max_scale=scale_max)
    mpe = multiscale_permutation_entropy(y, max_scale=scale_max)
    #--- plots
    axes[i+1].plot(range(1, scale_max + 1), mse, label=" MSE", marker='o')
    axes[i+1].plot(range(1, scale_max + 1), rcmse, label=" RCMSE", marker='o')
    axes[i+1].plot(range(1, scale_max + 1), mpe, label=" MPE", marker='o')
    axes[i+1].legend()

fig.tight_layout()


#------------------------ 
fig, axes = plt.subplots(3)
axes[0].pcolormesh(t_spectro, f_spectro, np.log(spectro + 0.00001), shading = 'nearest', cmap = 'rainbow', vmin = np.log(0.001), vmax = np.log(20))
for start in start_lists:
    axes[0].axvline(start, color = 'black')

for i in range(N_windows):
    
    y = windows[i]
    #--- compute multiscale entropy metrics
    mse = multiscale_entropy(y, max_scale=scale_max)
    mpe = multiscale_permutation_entropy(y, max_scale=scale_max)
    #--- plots
    axes[1].plot(range(1, scale_max + 1), mse, label=" MSE at " + str(start_lists[i]) + 's', marker='o')
    axes[2].plot(range(1, scale_max + 1), mse, label=" MPE at " + str(start_lists[i]) + 's', marker='o')

axes[1].legend()
axes[2].legend()

fig.tight_layout()

plt.show()




plt.show()



