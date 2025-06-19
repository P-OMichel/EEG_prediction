import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from Functions.entropy import multiscale_entropy, refined_composite_multiscale_entropy, multiscale_permutation_entropy
from Functions.open_file import open_clean_records

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
fig, (ax_signal, ax_spec, ax_output, ax_diff) = plt.subplots(4, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [1, 1.2, 1, 1]})
plt.subplots_adjust(left=0.1, bottom=0.25)

# Plot the full signal
signal_line, = ax_signal.plot(y, label="Full Signal")
vline_start = ax_signal.axvline(init_start, color='r', linestyle='--')
vline_end = ax_signal.axvline(init_start + init_size, color='r', linestyle='--')
ax_signal.set_title("Signal with Sliding Window")
ax_signal.set_xlim(0, len(y))

# Plot fixed spectrogram of the full signal
Pxx, freqs, bins, im = ax_spec.specgram(y, NFFT=256, Fs=fs, noverlap=240, cmap='rainbow')
ax_spec.set_title("Spectrogram")
ax_spec.set_ylabel("Frequency (Hz)")
ax_spec.set_xlabel("Time (samples)")

# Plot initial function outputs
output1 = multiscale_entropy(y[init_start:init_start + init_size], max_scale = scale_max)
output2 = multiscale_permutation_entropy(y[init_start:init_start + init_size], max_scale = scale_max)
line1, = ax_output.plot(range(1, scale_max + 1), output1, label='MSE')
line2, = ax_output.plot(range(1, scale_max + 1), output2, label='PME')
ax_output.set_title("Multiscale entropies on Current Window")
ax_output.set_ylim(0.3,2)
ax_output.legend()

# Plot initial difference 
line_diff, = ax_diff.plot(range(1, scale_max + 1), np.array(output1) - np.array(output2), label='MSE-PME')
ax_diff.set_title("Difference of Multiscale entropies on Current Window")
ax_diff.axhline(0, color ='orange')
ax_diff.set_ylim(-1,1)
ax_diff.legend()

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
    
    # Update vertical lines
    vline_start.set_xdata(start)
    vline_end.set_xdata(end)
    
    # Get current window
    window = y[start:end]

    # Recalculate outputs
    out1 = multiscale_entropy(window, max_scale = scale_max)
    out2 = multiscale_permutation_entropy(window, max_scale = scale_max)
    
    # Update lines
    line1.set_ydata(out1)
    line2.set_ydata(out2)
    line_diff.set_ydata(np.array(out1) - np.array(out2))
    
    fig.canvas.draw_idle()

# Connect sliders to update
slider_start.on_changed(update)
slider_size.on_changed(update)

plt.show()
