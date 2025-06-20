import numpy as np
from scipy.stats import entropy as shannon_entropy
from scipy.stats import gaussian_kde
from antropy import (
    sample_entropy,
    app_entropy,
    perm_entropy,
    spectral_entropy
)
from scipy.interpolate import interp1d
from scipy.integrate import simpson


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



#--------- extract metrics from multiscale entropy curve

def get_surface_linear(ME):

    y = np.array(ME)
    n = len(y)
    x = np.linspace(0, 1, n)  # Assume uniform spacing over [0, 1]

    # Line connecting first and last points
    slope = (y[-1] - y[0]) / (x[-1] - x[0])
    intercept = y[0]
    line = slope * x + intercept

    # Difference: (list - line)
    diff = y - line

    # Integral of the difference
    integral = np.trapz(diff, x)

    return integral 

def get_int_cumulative(ME):

    y = np.array(ME)
    n = len(y)
    x = np.linspace(0, 1, n)  # Assume uniform spacing over [0, 1]

    cumulative = np.cumsum(y)
    cumulative = cumulative / cumulative[-1]

    integral = np.trapz(x, cumulative)

    return integral


def get_int_cumulative_diff(ME0, ME1):

    y0 = np.array(ME0)
    y1 = np.array(ME1)
    y = y0 - y1
    n = len(y)
    x = np.linspace(0, 1, n)  # Assume uniform spacing over [0, 1]

    cumulative = np.cumsum(y)
    cumulative = cumulative / cumulative[-1]

    integral = np.trapz(cumulative, x)

    return integral

def analyze_curves(list1, list2, target_size=20):
    if len(list1) != len(list2):
        raise ValueError("Lists must have the same length.")
    if len(list1) >= target_size:
        raise ValueError("Original lists must be shorter than target size.")
    
    # Interpolation
    original_x = np.linspace(0, 1, len(list1))
    target_x = np.linspace(0, 1, target_size)
    interp1 = interp1d(original_x, list1, kind='linear')
    interp2 = interp1d(original_x, list2, kind='linear')
    y1 = interp1(target_x)
    y2 = interp2(target_x)
    
    # Difference
    diff = y1 - y2
    abs_diff = np.abs(diff)
    
    # Area between the curves
    area_between = simpson(abs_diff, x=target_x)
    integral_diff = np.trapz(diff, target_x)
    
    
    # Crossing point (first sign change)
    sign_diff = np.sign(diff)
    sign_changes = np.where(np.diff(sign_diff))[0]
    crossing_index = sign_changes[0] + 1 if len(sign_changes) > 0 else None
    crossing_position = target_x[crossing_index] if crossing_index is not None else None
    
    # Max difference info
    max_diff_index = np.argmax(abs_diff)
    max_diff_position = target_x[max_diff_index]
    
    # Center of mass of the difference curve (weighted average)
    center_of_mass_diff = np.sum(target_x * abs_diff) / np.sum(abs_diff)

    center_of_mass_index = np.argmin(np.abs(target_x - center_of_mass_diff))

    # Distance between center of mass and crossing point
    if crossing_position is not None:
        center_of_mass_vs_crossing = np.abs(center_of_mass_diff - crossing_position)
    else:
        center_of_mass_vs_crossing = None

    # Additional metrics
    mean_abs_diff = np.mean(abs_diff)
    max_diff_value = abs_diff[max_diff_index]
    above_ratio = np.sum(diff > 0) / target_size

    return {
        # 'interpolated_x': target_x,
        # 'interpolated_y1': y1,
        # 'interpolated_y2': y2,
        # 'difference': diff,
        'area_between': area_between,
        'integral_diff': integral_diff,
        'crossing_index': crossing_index,
        'crossing_position': crossing_position,
        'mean_absolute_difference': mean_abs_diff,
        'max_difference_value': max_diff_value,
        'max_difference_index': max_diff_index,
        'max_difference_position': max_diff_position,
        'center_of_mas_index': center_of_mass_index,
        'center_of_mass_diff': center_of_mass_diff,
        'center_of_mass_vs_crossing': center_of_mass_vs_crossing,
        'list1_mostly_above': above_ratio > 0.5,
    }
