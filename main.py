import numpy as np
import time
from scipy.optimize import linprog
import matplotlib.pyplot as plt
import os

# Create plots directory
os.makedirs("plots", exist_ok=True)

def generate_random_lp(n, m, seed=None):
    """
    Generate a random Linear Programming problem.
    min c^T x
    s.t. Ax <= b
    x >= 0
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate coefficients
    c = np.random.rand(n) * 10 - 5
    A = np.random.rand(m, n) * 10 - 5
    
    # Ensure feasible region exists (positive right hand side)
    x0 = np.random.rand(n) * 5
    b = np.dot(A, x0) + np.random.rand(m) * 5 
    
    return c, A, b

def run_experiment(method, n_list, m_list, num_trials=5, tol=1e-9):
    """
    Run experiments solving LP problems with increasing dimensions.
    """
    times = np.zeros((len(n_list), len(m_list)))
    iters = np.zeros((len(n_list), len(m_list)))
    
    for i, n in enumerate(n_list):
        for j, m in enumerate(m_list):
            t_total = 0
            iter_total = 0
            success_count = 0
            
            for k in range(num_trials):
                c, A, b = generate_random_lp(n, m, seed=i*1000 + j*100 + k)
                
                start_time = time.perf_counter()
                res = linprog(c, A_ub=A, b_ub=b, method=method, bounds=(0, None), 
                              options={'tol': tol, 'maxiter': 5000})
                elapsed = time.perf_counter() - start_time
                
                if res.success:
                    t_total += elapsed
                    if hasattr(res, 'nit'):
                        iter_total += res.nit
                    success_count += 1
            
            if success_count > 0:
                times[i, j] = t_total / success_count
                iters[i, j] = iter_total / success_count
            else:
                times[i, j] = np.nan
                iters[i, j] = np.nan
                
    return times, iters

def plot_metrics(x_list, metric_simp, metric_ip, xlabel, ylabel, title, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(x_list, metric_simp, marker='o', linewidth=2, label='Simplex (Revised)')
    plt.plot(x_list, metric_ip, marker='s', linewidth=2, label='Interior-Point')
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"plots/{filename}.png", dpi=300)
    plt.close()

def main():
    print("ME5414 Project: Comparing Simplex and Interior-Point Methods")
    print("Matriculation No.: A0326854W (Group B)\n")
    
    # 1. Experiment 1: Varying n with fixed m
    n_list_var = [50, 100, 200, 500, 1000]
    m_fixed = [50]
    
    print("Experiment 1: Varying n (50 to 1000), fixed m=50...")
    t_simp_n, i_simp_n = run_experiment('revised simplex', n_list_var, m_fixed)
    t_ip_n, i_ip_n = run_experiment('interior-point', n_list_var, m_fixed)
    
    plot_metrics(n_list_var, t_simp_n[:,0], t_ip_n[:,0], "Dimension of x (n)", "Computational Time (s)", 
                 "Computational Time vs Dimension n (fixed m=50)", "Time_vs_n")
    plot_metrics(n_list_var, i_simp_n[:,0], i_ip_n[:,0], "Dimension of x (n)", "Number of Iterations", 
                 "Iterations vs Dimension n (fixed m=50)", "Iters_vs_n")
    
    # 2. Experiment 2: Varying m with fixed n
    n_fixed = [50]
    m_list_var = [50, 100, 200, 500, 1000]
    
    print("Experiment 2: Varying m (50 to 1000), fixed n=50...")
    t_simp_m, i_simp_m = run_experiment('revised simplex', n_fixed, m_list_var)
    t_ip_m, i_ip_m = run_experiment('interior-point', n_fixed, m_list_var)
    
    plot_metrics(m_list_var, t_simp_m[0,:], t_ip_m[0,:], "Number of Inequalities (m)", "Computational Time (s)", 
                 "Computational Time vs Number of Inequalities m (fixed n=50)", "Time_vs_m")
    plot_metrics(m_list_var, i_simp_m[0,:], i_ip_m[0,:], "Number of Inequalities (m)", "Number of Iterations", 
                 "Iterations vs Number of Inequalities m (fixed n=50)", "Iters_vs_m")
    
    # 3. Experiment 3: Tolerance Sensitivity Check
    tol_list = [1e-4, 1e-6, 1e-8, 1e-10, 1e-12]
    t_simp_tol = []
    t_ip_tol = []
    i_simp_tol = []
    i_ip_tol = []
    
    n_tol, m_tol = 100, 100
    print(f"Experiment 3: Varying Tolerance (from 1e-4 to 1e-12), for n={n_tol}, m={m_tol}...")
    
    for tol in tol_list:
        ts, is_ = run_experiment('revised simplex', [n_tol], [m_tol], num_trials=3, tol=tol)
        ti, ii = run_experiment('interior-point', [n_tol], [m_tol], num_trials=3, tol=tol)
        t_simp_tol.append(ts[0,0])
        t_ip_tol.append(ti[0,0])
        i_simp_tol.append(is_[0,0])
        i_ip_tol.append(ii[0,0])
        
    # Plot tolerance with log scale for x-axis
    plt.figure(figsize=(10, 6))
    plt.plot(tol_list, t_simp_tol, marker='o', linewidth=2, label='Simplex (Revised)')
    plt.plot(tol_list, t_ip_tol, marker='s', linewidth=2, label='Interior-Point')
    plt.title(f"Computational Time vs Tolerance (fixed n={n_tol}, m={m_tol})", fontsize=14)
    plt.xlabel("Tolerance", fontsize=12)
    plt.ylabel("Computational Time (s)", fontsize=12)
    plt.xscale('log')
    plt.gca().invert_xaxis()  # smaller tolerance on the right
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("plots/Time_vs_Tolerance.png", dpi=300)
    plt.close()

    print("\nExperiments completed successfully. Plots saved in 'plots' folder.")

if __name__ == "__main__":
    main()