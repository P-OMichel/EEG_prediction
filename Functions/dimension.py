import numpy as np
from scipy.spatial.distance import pdist, squareform


# --- Correlation Dimension (Grassberger-Procaccia approximation) ---
def correlation_dimension(x, m=2, r=None):
    N = len(x) - m + 1
    if N <= 0:
        return np.nan
    embedded = np.array([x[i:i+m] for i in range(N)])
    dists = squareform(pdist(embedded, metric='chebyshev'))
    if r is None:
        r = np.std(x) * 0.2
    C = np.sum(dists < r) / (N**2)
    return np.log(C + 1e-10) / np.log(r + 1e-10)

# --- Hurst Exponent via Rescaled Range Analysis ---
def hurst_exponent(ts):
    N = len(ts)
    if N < 20:
        return np.nan
    X = np.cumsum(ts - np.mean(ts))
    R = np.max(X) - np.min(X)
    S = np.std(ts)
    if S == 0:
        return np.nan
    return np.log(R/S + 1e-10) / np.log(N + 1e-10)