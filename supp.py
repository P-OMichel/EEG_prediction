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
name = 'recordings/rec_20240321_085300.mat'
fs = 128
y, t = open_clean_records(name,fs,True,True) 

# Initial parameters
init_start = 0
init_size = int(fs * 30)

# parameters for thresholding
T = 0.12
q = 0.9

# Create figure and axes
fig, (ax_spec, ax_short_signal, ax_P, ax_P_spectro, ax_PCS) = plt.subplots(5, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [1.2, 1.2, 1, 1, 1]})
plt.subplots_adjust(left=0.1, bottom=0.25)

# Plot fixed spectrogram of the full signal
Pxx, freqs, bins, im = ax_spec.specgram(y, NFFT=256, Fs=fs, noverlap=240, cmap='rainbow')
vline_start = ax_spec.axvline(t[init_start], color='r', linestyle='--')
vline_end = ax_spec.axvline(t[init_start + init_size], color='r', linestyle='--')
ax_spec.set_title("Spectrogram")
ax_spec.set_ylabel("Frequency (Hz)")
ax_spec.set_xlabel("Time (samples)")

# plot initial window
y_w = t[init_start:init_start + init_size]
line_short_signal, = ax_short_signal.plot(y_w)
ax_short_signal.set_title("EEG in current window")
ax_short_signal.set_ylabel(r"Amplitude $\mu$V")
ax_short_signal.set_xlabel("Time (s)")
ax_short_signal.set_ylim(-75,75)

# Plot initial function outputs
P = P_y(y_w, fs, factor = 4)
line_P, = ax_P.plot(P)
h_line_P = ax_P.axhline(np.quantile(P, q) * T, color = 'orange')
ax_P.set_title("P")
ax_P.set_xlabel("Time (s)")
ax_P.set_ylim(0, 2)

t_spectro, f_spectro, spectro, P_spectro = P_spec(y_w, fs, delta_f = 1, n_overlap = 16, f_cut = 45)
line_P_spectro, = ax_P_spectro.plot(P_spectro)
h_line_P_spectro = ax_P_spectro.axhline(np.quantile(P_spectro, q) * T, color = 'orange')
ax_P_spectro.set_title("P_spectro")
ax_P_spectro.set_xlabel("Time (s)")
ax_P_spectro.set_ylim(0, 2)

env_int_PCS = int_PCS(f_spectro, spectro)
line_PCS, = ax_PCS.plot(env_int_PCS)
h_line_PCS = ax_PCS.axhline(np.quantile(env_int_PCS, q) * T, color = 'orange')
ax_PCS.set_title("int_PCS")
ax_PCS.set_xlabel("Time (s)")
ax_PCS.set_ylim(0, 2)

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
    
    # Update vertical lines (convert from samples to seconds if needed)
    vline_start.set_xdata(t[start])
    vline_end.set_xdata(t[end-1])

    # Get current window
    y_w = y[start:end]
    t_w = t[start:end]

    # Update short signal
    line_short_signal.set_data(np.arange(len(y_w)), y_w)
    ax_short_signal.set_xlim(0, len(y_w))

    # Recalculate outputs
    P = P_y(y_w, fs, factor=4)
    P /= np.quantile(P, q)
    line_P.set_data(np.arange(len(P)), P)
    ax_P.set_xlim(0, len(P))

    t_spectro, f_spectro, spectro, P_spectro = P_spec(y_w, fs, delta_f=1, n_overlap=16, f_cut=45)
    P_spectro /= np.quantile(P_spectro, q)
    line_P_spectro.set_data(np.arange(len(P_spectro)), P_spectro)
    ax_P_spectro.set_xlim(0, len(P_spectro))

    env_int_PCS = int_PCS(f_spectro, spectro)
    env_int_PCS /= np.quantile(env_int_PCS, q)
    line_PCS.set_data(np.arange(len(env_int_PCS)), env_int_PCS)
    ax_PCS.set_xlim(0, len(env_int_PCS))

    # Update horizontal threshold lines
    h_line_P.set_ydata(T)
    h_line_P_spectro.set_ydata(T)
    h_line_PCS.set_ydata(T)

    fig.canvas.draw_idle()

# Connect sliders to update
slider_start.on_changed(update)
slider_size.on_changed(update)

plt.show()
