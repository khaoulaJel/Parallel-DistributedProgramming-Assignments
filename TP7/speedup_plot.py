import matplotlib.pyplot as plt
import numpy as np

# Your measured data
np_list = np.array([1, 2, 4, 8, 16, 32])
times   = np.array([3.659782, 1.843046, 0.922247, 0.505736, 0.338127, 0.268407])

speedup    = times[0] / times
efficiency = speedup / np_list * 100

plt.figure(figsize=(10, 4.5))

plt.subplot(1, 2, 1)
plt.plot(np_list, speedup, 'o-', color='tab:blue', label='Measured speedup')
plt.plot(np_list, np_list, '--', color='gray', label='Ideal (linear)')
plt.xlabel('Number of processes')
plt.ylabel('Speedup')
plt.title('Speedup – Poisson 384×384')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(np_list, efficiency, 'o-', color='tab:orange', label='Efficiency')
plt.axhline(100, color='gray', linestyle='--', label='100%')
plt.xlabel('Number of processes')
plt.ylabel('Efficiency (%)')
plt.title('Parallel Efficiency')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('tp7_speedup_efficiency.png', dpi=150)
plt.show()