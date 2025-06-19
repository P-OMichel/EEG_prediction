import numpy as np
from scipy.stats import entropy as shannon_entropy
from scipy.stats import gaussian_kde
from antropy import (
    sample_entropy,
    app_entropy,
    perm_entropy,
    spectral_entropy
)


# Entropy functions
def shannon_entropy_fixed_bins(data, n_bins=10):
    hist, _ = np.histogram(data, bins=n_bins, density=True)
    hist = hist[hist > 0]
    return shannon_entropy(hist)

def shannon_entropy_kde(data, n_points=1000):
    kde = gaussian_kde(data)
    x_eval = np.linspace(min(data), max(data), n_points)
    pdf = kde.evaluate(x_eval)
    pdf = pdf / pdf.sum()
    return shannon_entropy(pdf)

def differential_entropy_kde(data, n_points=1000):
    kde = gaussian_kde(data)
    x_eval = np.linspace(min(data), max(data), n_points)
    pdf = kde.evaluate(x_eval)
    pdf = pdf / pdf.sum()
    return -np.sum(pdf * np.log(pdf + 1e-12))

def compute_approximate_entropy(data):
    return app_entropy(data, order=2, metric='chebyshev')

def compute_sample_entropy(data, order = 2):
    return sample_entropy(data, order=order, metric='chebyshev')

def compute_permutation_entropy(data):
    return perm_entropy(data, normalize=True)

def compute_spectral_entropy(data, sf=100):
    return spectral_entropy(data, sf=sf, method='fft', normalize=True)


#---------- Multiscale

def multiscale_entropy(signal, max_scale=10, m=2):
    """
    Compute Multiscale Entropy (MSE) for a 1D signal.

    Parameters:
        signal (array-like): 1D time series
        max_scale (int): Number of scales to compute
        m (int): Pattern length for Sample Entropy
        r (float or None): Tolerance (default: 0.2 * std at each scale)

    Returns:
        mse (list of float): Sample entropy at each scale
    """
    mse = []
    N = len(signal)
    for scale in range(1, max_scale + 1):
        # Step 1: Coarse-grain the signal at this scale
        num_segments = N // scale
        coarse = [np.mean(signal[i * scale:(i + 1) * scale]) for i in range(num_segments)]
        coarse = np.array(coarse)

        # Step 2: Compute Sample Entropy
        std = np.std(coarse)
        se = sample_entropy(coarse, order=m)
        mse.append(se)
    return mse

def refined_composite_multiscale_entropy(signal, max_scale=10, m=2):
    """
    Compute Refined Composite Multiscale Entropy (RCMSE) of a 1D signal.

    Parameters:
        signal (np.ndarray): 1D time series
        max_scale (int): Maximum scale τ
        m (int): Pattern length (embedding dimension)

    Returns:
        rcmse (list of float): RCMSE values at each scale
    """
    rcmse = []
    N = len(signal)

    for scale in range(1, max_scale + 1):
        entropy_values = []

        for offset in range(scale):
            # Create one of the scale-shifted coarse-grained signals
            truncated = signal[offset: N - ((N - offset) % scale)]
            if len(truncated) < scale * (m + 1):
                continue  # too short to compute entropy

            coarse = np.mean(truncated.reshape(-1, scale), axis=1)

            if len(coarse) > m + 1:
                entropy = sample_entropy(coarse, order=m)
                entropy_values.append(entropy)

        # Average entropy values over all shifts
        if len(entropy_values) > 0:
            rcmse.append(np.mean(entropy_values))
        else:
            rcmse.append(np.nan)

    return rcmse

def multiscale_permutation_entropy(signal, max_scale=10, order=3, delay=1):
    """
    Multiscale Permutation Entropy (MPE)
    """
    mpe = []
    N = len(signal)

    for scale in range(1, max_scale + 1):
        num_segments = N // scale
        if num_segments < order + 1:
            mpe.append(np.nan)
            continue
        coarse = np.mean(signal[:num_segments*scale].reshape(-1, scale), axis=1)
        pe = perm_entropy(coarse, order=order, delay=delay, normalize=True)
        mpe.append(pe)
    return mpe

