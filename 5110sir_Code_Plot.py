import numpy as np
import matplotlib.pyplot as plt
import random

def harmonic_sum(n):
    """Calculates the harmonic sum f(n) = sum(1/k) for k=1 to n."""
    return sum(1/k for k in range(1, n + 1))

def simulate_eradication_time(n):
    """
    Simulates the eradication time for n individuals starting in state II...I.
    Logic: The time to go from k infected to k-1 infected is exponentially 
    distributed with rate k (mean 1/k).
    """
    total_time = 0
    # Process goes from n infected down to 1 infected (then 0)
    for k in range(n, 0, -1):
        # random.expovariate(rate) returns a value from Exp(rate)
        time_step = random.expovariate(k)
        total_time += time_step
    return total_time

def main():
    # Parameters
    # We step by 10 or 20 to keep execution time reasonable while covering the range
    n_values = range(1, 1001, 10) 
    num_runs = 50  # "Averaged over many runs"
    
    simulated_means = []
    theoretical_values = []

    print(f"Running simulations for n = 1 to 1000 ({num_runs} runs per n)...")

    for n in n_values:
        # 1. Theoretical Calculation: f(n)
        theoretical_values.append(harmonic_sum(n))
        
        # 2. Simulation: Run 'num_runs' times and average
        runs = [simulate_eradication_time(n) for _ in range(num_runs)]
        avg_time = sum(runs) / len(runs)
        simulated_means.append(avg_time)

    # Plotting
    plt.figure(figsize=(10, 6))
    
    # Plot Simulation
    plt.plot(n_values, simulated_means, 'o', markersize=2, label='Simulated Avg T', alpha=0.6, color='blue')
    
    # Plot Theory
    plt.plot(n_values, theoretical_values, '-', linewidth=2, label='Theoretical f(n)', color='red')
    
    plt.title('Eradication Time: Simulation vs. Harmonic Sum f(n)')
    plt.xlabel('Number of Individuals (n)')
    plt.ylabel('Expected Time (Weeks)')
    plt.legend()
    plt.grid(True)
    
    # Save or Show
    print("Simulation complete. Displaying plot.")
    plt.show()

if __name__ == "__main__":
    main()
