import numpy as np
import hfda
import matplotlib.pyplot as plt

# Custom Higuchi Fractal Dimension function
def higuchi_fd(signal, k_max):
    N = len(signal)
    L = []
    k_values = np.arange(1, k_max + 1)

    for k in k_values:
        Lk = []
        for m in range(k):
            idxs = np.arange(1, int(np.floor((N - m) / k)), dtype=int)
            if len(idxs) == 0:
                continue
            diffs = np.abs(signal[m + idxs * k] - signal[m + k * (idxs - 1)])
            lm = np.sum(diffs)
            norm_factor = (N - 1) / (len(idxs) * k)
            Lm = (lm * norm_factor) / k
            Lk.append(Lm)
        if len(Lk) > 0:
            L.append(np.mean(Lk))
        else:
            L.append(0)

    log_k = np.log(1 / k_values)
    log_L = np.log(L)
    coeffs = np.polyfit(log_k, log_L, 1)
    return -coeffs[0]  # The slope is the fractal dimension

# Generate a noisy sine wave signal
np.random.seed(42)
t = np.linspace(0, 6 * np.pi, 1000)
signal = np.sin(t) + 0.5 * np.random.randn(1000)

# Parameters
k_max = 10

# Compute HFD using custom function
fd_custom = higuchi_fd(signal, k_max)

# Compute HFD using hfda package
fd_hfda = hfda.measure(signal, k_max)

print(f"Custom HFD: {fd_custom:.4f}")
print(f"Hfda HFD:   {fd_hfda:.4f}")
