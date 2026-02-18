N = 40000  
M = 600    
TOTAL_FLOPS = 2.0 * N * M  # One multiply and one add per inner loop iteration

threads = [1, 2, 4, 8, 16]
v1_times = [0.025120, 0.020921, 0.012878, 0.007628, 0.004436]
v2_times = [0.025481, 0.018460, 0.011270, 0.007245, 0.005515]
v3_times = [0.023508, 0.018151, 0.013112, 0.007491, 0.004271]

def print_metrics(name, times):
    print(f"\n--- Metrics for {name} ---")
    print(f"{'Threads':<8} | {'Time (s)':<10} | {'MFLOP/s':<12} | {'Speedup':<8} | {'Efficiency (%)':<15}")
    print("-" * 65)
    
    t1 = times[0]
    for i in range(len(threads)):
        t_p = times[i]
        p = threads[i]
        
        mflops = TOTAL_FLOPS / (t_p * 1e6)
        speedup = t1 / t_p
        efficiency = (speedup / p) * 100
        
        print(f"{p:<8} | {t_p:<10.6f} | {mflops:<12.2f} | {speedup:<8.2f} | {efficiency:<15.2f}")

print_metrics("Version 1 (Implicit Barrier)", v1_times)
print_metrics("Version 2 (Dynamic + Nowait)", v2_times)
print_metrics("Version 3 (Static + Nowait)", v3_times)