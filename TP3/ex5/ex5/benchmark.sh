#!/bin/bash

# Compile the program
echo "Compiling jacobi_optimized.c..."
gcc -fopenmp -O3 -o jacobi_optimized jacobi_optimized.c -lm

if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi

echo "Compilation successful!"
echo ""

# Array of thread counts
threads=(1 2 4 8 16)

# Number of runs for averaging
NUM_RUNS=3

# Output file for results
output_file="jacobi_results.csv"
echo "Threads,Elapsed_Time,Speedup,Efficiency" > $output_file

echo "Running benchmarks..."
echo "===================="

# Function to get average time
get_average_time() {
    local num_threads=$1
    local total_time=0
    
    for ((i=1; i<=NUM_RUNS; i++)); do
        result=$(./jacobi_optimized $num_threads | cut -d',' -f2)
        total_time=$(echo "$total_time + $result" | bc -l)
    done
    
    avg=$(echo "scale=6; $total_time / $NUM_RUNS" | bc -l)
    echo $avg
}

# Get baseline (serial) time
echo "Running with 1 thread (serial)..."
serial_time=$(get_average_time 1)
echo "Serial time: $serial_time sec"
echo ""

# Run for each thread count
for t in "${threads[@]}"
do
    echo "Testing with $t threads..."
    
    # Get average elapsed time
    elapsed=$(get_average_time $t)
    
    # Calculate speedup
    speedup=$(echo "scale=4; $serial_time / $elapsed" | bc -l)
    
    # Calculate efficiency (as percentage)
    efficiency=$(echo "scale=2; ($speedup / $t) * 100" | bc -l)
    
    echo "  Time: $elapsed sec"
    echo "  Speedup: $speedup"
    echo "  Efficiency: $efficiency%"
    echo ""
    
    # Save to file
    echo "$t,$elapsed,$speedup,$efficiency" >> $output_file
done

echo "Results saved to $output_file"
