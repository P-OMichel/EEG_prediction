'''
Function to detect burst between suppressions
'''
import numpy as np
import scipy as sc
from Functions.utils import filter_binary_mask, remove_short_segments

def get_burst(edge_frequency, delta_t):

    N = len(edge_frequency)

    # get mask of values bellow 8
    mask_8 = np.zeros(N)
    mask_8[np.where(edge_frequency<=8)[0]] = 1
    mask_8 = sc.ndimage.binary_closing(mask_8,np.ones(int(0.5 / delta_t)))

    # initialize mask for burst
    mask_burst = np.zeros(N)

    # check if presence of supp to check for presence of burst
    if np.sum(mask_8) >= N/3:
        mask_burst = ((edge_frequency >= 15) &  (edge_frequency <= 40)).astype(int)
        mask_burst = filter_binary_mask(mask_burst, int(1 / delta_t))

    return mask_burst