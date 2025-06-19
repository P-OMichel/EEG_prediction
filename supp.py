import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from Functions.open_file import open_clean_records
from Functions.time_frequency import spectrogram

def P_y(y, fs, factor = 4):
    N = int(fs / factor)
    h = np.ones(N) / N
    y2 = np.convolve(y**2, h, mode = 'same')
    y2 = np.convolve(y2, h, mode = 'same')
    return y2

def P_spec(y, fs, delta_f=1, n_overlap=16, f_cut=45):
    t_spectro, f_spectro, spectro = spectrogram(y,fs, delta_f, n_overlap, f_cut)
    P_spectro = np.sum(spectro, axis = 0)
    return t_spectro, f_spectro, spectro, P_spectro

def int_PCS(f_spectro, spectro):
    PCS = np.cumsum(spectro, axis = 0)
    env_int_PCS = np.trapz(PCS, f_spectro, axis = 0)
    return env_int_PCS

#--- load data
name = 'recordings/rec_20240307_115040.mat'
fs = 128
y, t = open_clean_records(name,fs,True,True) 

# Initial parameters
init_start = 0
init_size = int(fs * 30)

# parameter for multiscale entropy
scale_max = 5

# Create figure and axes
fig, (ax_spec, ax_short_signal, ax_short_spectro, ax_P, ax_P_spectro, ax_PCS) = plt.subplots(6, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [1.2, 1, 1, 1, 1, 1]})
plt.subplots_adjust(left=0.1, bottom=0.25)

# Plot fixed spectrogram of the full signal
Pxx, freqs, bins, im = ax_spec.specgram(y, NFFT=256, Fs=fs, noverlap=240, cmap='rainbow')
vline_start = ax_spec.axvline(init_start, color='r', linestyle='--')
vline_end = ax_spec.axvline(init_start + init_size, color='r', linestyle='--')
ax_spec.set_title("Spectrogram")
ax_spec.set_ylabel("Frequency (Hz)")
ax_spec.set_xlabel("Time (samples)")

# plot initial window
t_w, y_w = t[init_start:init_start + init_size], y[init_start:init_start + init_size]
line_short_signal, = ax_short_signal.plot(t_w, y_w)
ax_short_signal.set_title("EEG in current window")
ax_short_signal.set_ylabel(r"Amplitude $\mu$V")
ax_short_signal.set_xlabel("Time (s)")

# Plot initial function outputs
P = P_y(y_w, fs, factor = 4)
line_P, = ax_P.semilogy(t_w, P)
ax_P.set_title("P")
ax_P.set_xlabel("Time (s)")

t_spectro, f_spectro, spectro, P_spectro = P_spec(y_w, fs, delta_f = 1, n_overlap = 16, f_cut = 45)
line_P_spectro, = ax_P_spectro.semilogy(t_spectro + t_w[0], P_spectro)
ax_P_spectro.set_title("P_spectro")
ax_P_spectro.set_xlabel("Time (s)")

env_int_PCS = int_PCS(f_spectro, spectro)
line_PCS, = ax_PCS.semilogy(t_spectro + t_w[0], env_int_PCS)
ax_PCS.set_title("int_PCS")
ax_PCS.set_xlabel("Time (s)")

# Create sliders
axcolor = 'lightgoldenrodyellow'
ax_start = plt.axes([0.1, 0.15, 0.8, 0.03], facecolor=axcolor)
ax_size = plt.axes([0.1, 0.1, 0.8, 0.03], facecolor=axcolor)

slider_start = Slider(ax_start, 'Start', 0, len(y) - 10, valinit=init_start, valstep=1)
slider_size = Slider(ax_size, 'Window Size', 10, 12800, valinit=init_size, valstep=1)

# Update function
def update(val):
    start = int(slider_start.val)
    size = int(slider_size.val)
    end = min(start + size, len(y))

    print(start, end) 
    
    # Update vertical lines
    # vline_start.set_xdata(start)
    # vline_end.set_xdata(end)
    
    # Get current window
    y_w = y[start:end]
    t_w = t[start:end]

    # Recalculate outputs
    P = P_y(y_w, fs, factor = 4)
    t_spectro, f_spectro, spectro, P_spectro = P_spec(y_w, fs, delta_f = 1, n_overlap = 16, f_cut = 45)
    env_int_PCS = int_PCS(f_spectro, spectro)
    
    # plot new data
    line_short_signal.set_ydata(y_w) #t_w, 
    line_P.set_ydata(P) #t_w, 
    line_P_spectro.set_ydata(P_spectro) #t_spectro + t_w[0], 
    line_PCS.set_ydata(env_int_PCS) #t_spectro + t_w[0], 

    fig.canvas.draw_idle()

# Connect sliders to update
slider_start.on_changed(update)
slider_size.on_changed(update)

plt.show()
