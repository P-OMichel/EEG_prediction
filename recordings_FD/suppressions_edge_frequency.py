'''
File to detect the suppressions using a TF representation to extract the edge frequency
'''
import numpy as np
import matplotlib.pyplot as plt
import Functions.time_frequency as tf
from Functions.suppressions import detect_supp_edge_frequency, detection_supp_high_prominence, supp_edge_frequency_prominence, detect_supp_P_spectro, detect_suppressions_power, detect_sup_amplitude

plt.rcParams['font.size'] = 16
plt.rcParams['font.family'] = 'Times New Roman'

#--- load data
y = np.load('recordings_FD/LFP_25-01-10_15-32-19.npy')
q = np.quantile(y,0.9)
y = y / 40
factor = 10
y = y[::factor]
fs = int(2500 / factor)
N = len(y)
t = np.linspace(0, N/fs, N)


Ws = 45 * fs
windows = [y[i:i+Ws] for i in range(0,len(y)-Ws,Ws)]

fig, axes = plt.subplots(7, sharex=True, gridspec_kw={'height_ratios': [3,4,2,2,2,2,2]})
axes[0].set_ylim(-75, 75)
t0 = 0

med_edge_frequency_lv = []
med_edge_frequency_sv = []

titles = ['EEG signal', 'Spectrogram | edge frequencies (black)', 'IES mask', r'$\alpha$-suppressions mask', r'$\alpha$-suppressions mask (prominence)', 'Threshold value', 'Spectrogram density sum']

for i in range(len(windows)):
    N = len(windows[i])
    print(N)
    t = t0 + np.linspace(0, N / fs, N)

    # get spectrogram
    t_spectro, f_spectro, spectro = tf.spectrogram(windows[i],fs)
    delta_t = t_spectro[1] - t_spectro[0]
    Fs = 1 / delta_t
    # get edge_frequency using limit_value
    edge_frequency_lv, edge_frequencies_hf, threshold_lv = tf.edge_frequencies_limit_value(spectro,f_spectro,T=10)
    # get supp for edge_frequency_lv
    mask_IES_lv, mask_alpha_lv, mask_alpha_minus_IES_lv, IES_proportion_lv, alpha_suppression_proportion_lv = detect_supp_edge_frequency(edge_frequency_lv, Fs)
    # get supp prominence on edge_frequencies
    mask_prom = supp_edge_frequency_prominence(edge_frequency_lv, Fs)
    # get mask by computing power of spectrogram within a frequency range
    mask_P_spectro, P_spectro, threshold_P_spectro = detect_supp_P_spectro(edge_frequency_lv, spectro, f_spectro)

    # check if in a zone of high alpha supp
    if alpha_suppression_proportion_lv > 0.3 :
        edge_frequency_lv, edge_frequencies_hf, threshold_lv = tf.edge_frequencies_limit_value(spectro,f_spectro, T = 2, q = 0.95)
        mask_IES_lv, mask_alpha_lv, mask_alpha_minus_IES_lv, IES_proportion_lv, alpha_suppression_proportion_lv = detect_supp_edge_frequency(edge_frequency_lv, Fs)

    # get supp prominence for edge_frequency_lv
    pos_min_lv, mask_supp_lv = detection_supp_high_prominence(edge_frequency_lv, delta_t)[:2]

    # get supp power
    mask_IES_P, mask_alpha_P = detect_suppressions_power(windows[i], fs, 12, 5)[5:7]
    # get supp amplitude
    #mask_IES_amplitude, mask_alpha_amplitude = detect_sup_amplitude(t, windows[i])[:2]

    # update time for current iteration
    t_spectro = np.linspace(t[0], t[-1], len(t_spectro))

    axes[0].plot(t, windows[i] - np.mean(windows[i]), color = 'blue')

    axes[1].pcolormesh(t_spectro, f_spectro, np.log(spectro + 0.000000001), shading='nearest', cmap='jet', vmin=np.log(0.001), vmax=np.log(20))
    axes[1].plot(t_spectro, edge_frequency_lv, color = 'black')
    axes[2].plot(t_spectro, mask_IES_lv, color = 'red', linestyle = '--')
    axes[3].plot(t_spectro, mask_alpha_lv, color = 'red', linestyle = '--')
    for pos in pos_min_lv:
        axes[1].plot(t_spectro[pos], edge_frequency_lv[pos], 'o', color = 'black')
    axes[4].plot(t_spectro, mask_supp_lv, color = 'red', linestyle = '--')
    axes[5].plot(t[-1], threshold_lv, 'o', color='red')

    axes[3].plot(t_spectro, mask_P_spectro, color = 'green', alpha = 0.5)



    axes[2].plot(t, mask_IES_P, color = 'black', alpha = 0.5)
    axes[3].plot(t, mask_alpha_P, color = 'black', alpha = 0.5)

    # axes[2].plot(t, mask_IES_amplitude, color = 'blue', alpha = 0.5)
    # axes[3].plot(t, mask_alpha_amplitude, color = 'blue', alpha = 0.5)
    
    
    axes[6].semilogy(t_spectro, P_spectro, color = 'blue')
    axes[6].semilogy(t_spectro, [threshold_P_spectro] * len(P_spectro), color = 'orange')

    # update time marker
    t0 = t[-1]

    for i in range(7):
        axes[i].set_title(titles[i])

fig.tight_layout()
plt.show()
