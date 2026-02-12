import pandas as pd
import matplotlib.pyplot as plt

# Load benchmark data
df = pd.read_csv("jacobi_results.csv")

threads = df['Threads']
time = df['Elapsed_Time']    # matches your CSV
speedup = df['Speedup']
efficiency = df['Efficiency']

# -------- Plot Performance (Time vs Threads) --------
plt.figure(figsize=(8,5))
plt.plot(threads, time, marker='o', linestyle='-', color='blue')
plt.xlabel("Number of Threads")
plt.ylabel("Execution Time (s)")
plt.title("Jacobi Method Performance")
plt.grid(True)
plt.xticks(threads)
plt.savefig("jacobi_time.png")
plt.show()

# -------- Plot Scalability (Speedup vs Threads) --------
plt.figure(figsize=(8,5))
plt.plot(threads, speedup, marker='o', linestyle='-', color='green', label='Speedup')
plt.plot(threads, threads, linestyle='--', color='red', label='Ideal Speedup')
plt.xlabel("Number of Threads")
plt.ylabel("Speedup")
plt.title("Jacobi Method Scalability")
plt.grid(True)
plt.xticks(threads)
plt.legend()
plt.savefig("jacobi_speedup.png")
plt.show()

# -------- Plot Efficiency (Efficiency vs Threads) --------
plt.figure(figsize=(8,5))
plt.plot(threads, efficiency, marker='o', linestyle='-', color='purple')
plt.xlabel("Number of Threads")
plt.ylabel("Efficiency (%)")
plt.title("Jacobi Method Efficiency")
plt.grid(True)
plt.xticks(threads)
plt.savefig("jacobi_efficiency.png")
plt.show()
