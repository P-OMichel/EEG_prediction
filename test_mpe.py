import numpy as np
import matplotlib.pyplot as plt
from Functions.entropy import multiscale_entropy, refined_composite_multiscale_entropy, multiscale_permutation_entropy
from Functions.open_file import open_clean_records
import time 

#--- load data
name = 'recordings/rec_20240321_085300.mat'
fs = 128
signal, t = open_clean_records(name,fs,True,True) 
signal[1200*fs:1300*fs]

scale_max = 5

# Compute MPE
t0 = time.time()
mpe = multiscale_permutation_entropy(signal, max_scale=scale_max)
t1 = time.time()
print('time to compute MPE:', t1 - t0)

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(range(1, scale_max + 1), mpe, label=" MPE", marker='o')
plt.xlabel("Scale (τ)")
plt.ylabel("Multiscale Entropy")
plt.title("Multiscale Entropy Curves")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
