import math                       
import random                     
from typing import List, Tuple    
import numpy as np                
import matplotlib.pyplot as plt   

# =============================================================================
# Gillespie Algorithm Implemented
# =============================================================================

# S=0, I=1, R=2
S = 0  # （Susceptible）
I = 1  # （Infected）
R = 2  # （Recovered）

def harmonic_sum(n: int) -> float:
    """
    f(n) = sum_{k=1}^n 1/k
    """
    total = 0.0                       
    for k in range(1, n + 1):         
        total += 1.0 / k              
    return total                  

def build_neighbor_indices(n: int) -> List[Tuple[int, int]]:

    neighbors = []                    
    for i in range(n):                
        left = (i - 1) % n            
        right = (i + 1) % n            
        neighbors.append((left, right))
    return neighbors                   

def sample_exponential(rate: float, rng: random.Random) -> float:
    u = rng.random()                   
    return -math.log(1.0 - u) / rate   

def simulate_one_run(
    n: int,
    lam: float,
    rng: random.Random,
    initial_state: List[int],
    track_ever_infected: bool = False,
) -> Tuple[float, int]:
    neighbors = build_neighbor_indices(n)  

    state = list(initial_state)           

    t = 0.0                               

    ever_infected = set()                 
    if track_ever_infected:                
        for i, s in enumerate(state):       
            if s == I:                     
                ever_infected.add(i)        

    while True:                          
        infected_indices = [             
            i for i, s in enumerate(state) if s == I
        ]

        if not infected_indices:       
            T = t                     
            M = len(ever_infected) if track_ever_infected else 0 
            return T, M                  

        events = []                     
        total_rate = 0.0                 

        for i in infected_indices:   
            rate = 1.0                  
            events.append((rate, "rec", i, None))  
            total_rate += rate            

        for i in infected_indices:        
            left, right = neighbors[i]     

            if state[left] == S:           
                rate = lam                  
                events.append((rate, "inf", i, left))  
                total_rate += rate          

            if state[right] == S:           
                rate = lam               
                events.append((rate, "inf", i, right)) 
                total_rate += rate        

        if total_rate <= 0.0:             
            T = t                          
            M = len(ever_infected) if track_ever_infected else 0
            return T, M                     

        dt = sample_exponential(total_rate, rng)  
        t += dt                                   

        threshold = rng.random() * total_rate     
        accum = 0.0                               
        chosen_event = None                      

        for (rate, kind, i, j) in events:       
            accum += rate                        
            if accum >= threshold:              
                chosen_event = (kind, i, j)      
                break
            
        if chosen_event is None:                 
            T = t                                
            M = len(ever_infected) if track_ever_infected else 0
            return T, M

        kind, i, j = chosen_event                

        if kind == "rec":                         
            if state[i] == I:                    
                state[i] = R                 

        elif kind == "inf":                   
            if state[j] == S:                  
                state[j] = I                  
                if track_ever_infected:        
                    ever_infected.add(j)      



def simulate_eradication_all_infected(
    n: int,
    lam: float,
    num_runs: int,
    seed: int = 0,
) -> List[float]:
    rng = random.Random(seed)               
    Ts = []                                
    initial_state = [I] * n              

    for _ in range(num_runs):               
        T, _ = simulate_one_run(            
            n=n,
            lam=lam,
            rng=rng,
            initial_state=initial_state,
            track_ever_infected=False,       
        )
        Ts.append(T)                       

    return Ts                              

def run_experiment_E_T_equals_f_n(
    n_values: List[int],
    lam: float = 1.0,
    num_runs: int = 200,
):
    avg_Ts = []                             
    f_values = []                            

    for n in n_values:                     
        Ts = simulate_eradication_all_infected(
            n=n,
            lam=lam,                         
            num_runs=num_runs,              
            seed=42 + n,                     
        )
        avg_T = sum(Ts) / len(Ts)          
        avg_Ts.append(avg_T)                 
        f_values.append(harmonic_sum(n))     

    plt.figure()                             
    plt.plot(n_values, avg_Ts, label="Average T (simulation)")  
    plt.plot(n_values, f_values, label="f(n) = sum 1/k")        
    plt.xlabel("n (number of individuals)")  # x
    plt.ylabel("Time")                       # y
    plt.title("Eradication Time vs Harmonic Sum")  
    plt.legend()                             
    plt.grid(True)                          
    plt.tight_layout()                       
    plt.show()                             

def build_initial_states_n5_patient_zero(
    scenario: str,
    rng: random.Random,
) -> List[int]:
    """
    n=5:
      1) baseline;
      2) neighbor;
      3) random
    """
    n = 5                                    
    state = [S] * n                          
    patient_zero = 0                      
    state[patient_zero] = I                
    # ---------- baseline ----------
    if scenario == "baseline":              
        return state                        
    # ---------- neighbor ----------
    elif scenario == "neighbor":             
        inoculated_neighbor = 1              
        state[inoculated_neighbor] = R       
        return state                       
    # ---------- random ----------
    elif scenario == "random":              
        candidates = [2, 3]                  
        inoculated = rng.choice(candidates) 
        state[inoculated] = R               
        return state                     

    else:
        raise ValueError("Unknown scenario: " + scenario) 

def simulate_M_vs_lambda(
    lam_values: List[float],
    num_runs: int = 500,
    seed: int = 0,
):
    """
    n=5:
      1) baseline;
      2) neighbor;
      3) random
    """
    n = 5                                    
    rng_global = random.Random(seed)        

    avg_M_baseline = []                     
    avg_M_neighbor = []                      
    avg_M_random = []                        

    for lam in lam_values:                  
        Ms_baseline = []                
        Ms_neighbor = []                    
        Ms_random = []                       
        for run_idx in range(num_runs):     
            rng = random.Random(seed + run_idx)  
            # ---------- baseline ----------
            init_baseline = build_initial_states_n5_patient_zero(
                scenario="baseline",
                rng=rng_global,              
            )
            T_b, M_b = simulate_one_run(    
                n=n,
                lam=lam,
                rng=rng,
                initial_state=init_baseline,
                track_ever_infected=True,   
            )
            Ms_baseline.append(M_b)         

            # ---------- neighbor ----------
            init_neighbor = build_initial_states_n5_patient_zero(
                scenario="neighbor",
                rng=rng_global,           
            )
            T_n, M_n = simulate_one_run(     
                n=n,
                lam=lam,
                rng=rng,
                initial_state=init_neighbor,
                track_ever_infected=True,
            )
            Ms_neighbor.append(M_n)       
            # ---------- random ----------
            init_random = build_initial_states_n5_patient_zero(
                scenario="random",
                rng=rng_global,             
            )
            T_r, M_r = simulate_one_run(    
                n=n,
                lam=lam,
                rng=rng,
                initial_state=init_random,
                track_ever_infected=True,
            )
            Ms_random.append(M_r)            

        avg_M_baseline.append(sum(Ms_baseline) / len(Ms_baseline))
        avg_M_neighbor.append(sum(Ms_neighbor) / len(Ms_neighbor))
        avg_M_random.append(sum(Ms_random) / len(Ms_random))

    plt.figure()                            
    plt.plot(lam_values, avg_M_baseline, label="Baseline (no inoculation)")
    plt.plot(lam_values, avg_M_neighbor, label="Inoculate neighbor")
    plt.plot(lam_values, avg_M_random, label="Inoculate random non-neighbor")
    plt.xscale("log")                        
    plt.xlabel("λ (infection rate)")             # x
    plt.ylabel("E[M] (average total infected)")  # y
    plt.title("Impact of Inoculation Strategy on Total Infections (n=5)") 
    plt.legend()                           
    plt.grid(True)                          
    plt.tight_layout()                      
    plt.show()                              

if __name__ == "__main__":
    # Part 1：E[T] ≈ f(n)
    n_values = list(range(1, 51))           
    run_experiment_E_T_equals_f_n(
        n_values=n_values,
        lam=1.0,                            
        num_runs=200,                      
    )

    # -------------------------------------------------------------------------
    # Part 2：n=5
    lam_values = np.logspace(-1, math.log10(200.0), 15)  
    simulate_M_vs_lambda(
        lam_values=lam_values.tolist(),     
        num_runs=500,                      
        seed=123,                          
    )
