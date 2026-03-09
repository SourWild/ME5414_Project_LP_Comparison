"""
ME5414 Project: Comparison of Simplex and Interior-Point Methods for Linear Programming
Author: A0326854W
Group: B (Simplex method vs Interior-Point method)

Problem Group B:
- Non-interior-point method: Simplex Algorithm (specifically Revised Simplex via HiGHS dual simplex)
- Interior-point method: Interior-Point (IPM) via HiGHS IPM solver

Comparison is done on:
  1. Computational time vs. dimension n (fixed m)
  2. Computational time vs. number of inequalities m (fixed n)
  3. Iterations vs. n, vs. m
  4. Sensitivity to stopping tolerance
"""

import numpy as np
import time
from scipy.optimize import linprog
import matplotlib.pyplot as plt
import matplotlib
import os
import warnings

warnings.filterwarnings("ignore")  # suppress deprecation warnings
matplotlib.use('Agg')  # non-interactive backend

# Create plots directory
os.makedirs("plots", exist_ok=True)

# -------------------
# Problem Generator
# -------------------
def generate_random_lp(n, m, seed=None):
    """
    Generate a random Linear Programming problem:
        min c^T x
        s.t. A x <= b
             x >= 0
             x <= 10   (upper bounds added as inequality constraints to ensure boundedness)

    Guarantees:
     - A strictly feasible interior point exists.
     - The feasible region is bounded (via simple upper bounds on x).

    Args:
        n: Number of decision variables (dimension of x)
        m: Number of inequality constraints
        seed: Random seed for reproducibility
    Returns:
        c (n,): Cost vector
        A_full (m+n, n): Constraint matrix (original + upper bounds)
        b_full (m+n,): Right-hand side vector
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Random cost vector
    c = np.random.randn(n)

    # Generate a strictly feasible interior point x0 in (0, 10)
    x0 = np.random.rand(n) * 5 + 2.0  # x0 in [2, 7]

    # Random constraint matrix
    A = np.random.randn(m, n)

    # Set b such that A x0 + slack = b where slack > 0
    slack = np.random.rand(m) * 5 + 1.0
    b = A @ x0 + slack

    # Add upper bound constraints x_i <= 100 to guarantee boundedness
    A_ub = np.eye(n)
    b_ub = np.full(n, 100.0)

    A_full = np.vstack([A, A_ub])
    b_full = np.concatenate([b, b_ub])

    return c, A_full, b_full


# -------------------
# Experiment Runner
# -------------------
def solve_lp(c, A, b, method, tol):
    """
    Solve LP using scipy's linprog with the specified HiGHS method.

    Args:
        c, A, b: LP problem definition
        method: 'highs-ds' for simplex, 'highs-ipm' for interior-point
        tol: Stopping tolerance (dual feasibility tolerance)
    Returns:
        success: Whether the solver succeeded
        elapsed: Elapsed wall-clock time in seconds
        n_iter: Number of iterations taken
    """
    options = {
        'dual_feasibility_tolerance': tol,
        'primal_feasibility_tolerance': tol,
        'ipm_optimality_tolerance': tol,
    }
    start = time.perf_counter()
    res = linprog(c, A_ub=A, b_ub=b, method=method, bounds=(0, None), options=options)
    elapsed = time.perf_counter() - start

    success = res.success
    n_iter = res.get('message', '')  # nit not reliably provided, use 0
    n_iter = res.nit if hasattr(res, 'nit') and res.nit is not None else 0

    return success, elapsed, n_iter


def run_experiment(method, n_list, m_list, num_trials=5, tol=1e-8):
    """
    Run experiments over all (n, m) combinations.
    
    Returns:
        times (len(n_list), len(m_list)): Average wall-clock time per solve
        iters (len(n_list), len(m_list)): Average number of iterations
    """
    times = np.zeros((len(n_list), len(m_list)))
    iters = np.zeros((len(n_list), len(m_list)))
    
    for i, n in enumerate(n_list):
        for j, m in enumerate(m_list):
            t_list, it_list = [], []
            
            for k in range(num_trials):
                c, A, b = generate_random_lp(n, m, seed=i*1000 + j*100 + k)
                success, elapsed, n_iter = solve_lp(c, A, b, method, tol)
                
                if success:
                    t_list.append(elapsed)
                    it_list.append(n_iter)
            
            times[i, j] = np.mean(t_list) if t_list else np.nan
            iters[i, j] = np.mean(it_list) if it_list else np.nan
    
    return times, iters


# -------------------
# Plotting Helper
# -------------------
def plot_comparison(x_list, y_simp, y_ip, xlabel, ylabel, title, filename):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_list, y_simp, marker='o', linewidth=2, markersize=8, label='Simplex (highs-ds)')
    ax.plot(x_list, y_ip, marker='s', linewidth=2, markersize=8, label='Interior-Point (highs-ipm)')
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"plots/{filename}.png", dpi=150)
    plt.close()


# -------------------
# Main Routine
# -------------------
def main():
    print("=" * 60)
    print("ME5414 Project: LP Solver Comparison")
    print("Matriculation No.: A0326854W  |  Group B")
    print("Simplex (highs-ds) vs Interior-Point (highs-ipm)")
    print("=" * 60)

    TOLERANCE = 1e-8

    # -----------------------------------------------
    # Experiment 1: Varying n (dimension of x), fixed m
    # -----------------------------------------------
    n_list = [50, 100, 200, 500, 1000]
    m_fixed = [100]

    print(f"\n[Experiment 1] Varying n, fixed m={m_fixed[0]}, tolerance={TOLERANCE}")
    t_simp_n, i_simp_n = run_experiment('highs-ds', n_list, m_fixed, num_trials=5, tol=TOLERANCE)
    t_ip_n, i_ip_n   = run_experiment('highs-ipm', n_list, m_fixed, num_trials=5, tol=TOLERANCE)

    plot_comparison(n_list, t_simp_n[:,0], t_ip_n[:,0],
                    "Dimension of Decision Variable (n)", "Mean Computational Time (s)",
                    f"Computational Time vs n (fixed m={m_fixed[0]})",
                    "Time_vs_n")
    plot_comparison(n_list, i_simp_n[:,0], i_ip_n[:,0],
                    "Dimension of Decision Variable (n)", "Mean Number of Iterations",
                    f"Iterations vs n (fixed m={m_fixed[0]})",
                    "Iters_vs_n")

    print("  n_list:", n_list)
    print("  Simplex times:",  np.round(t_simp_n[:,0], 6))
    print("  IP      times:",  np.round(t_ip_n[:,0], 6))
    print("  Simplex iters:",  np.round(i_simp_n[:,0], 1))
    print("  IP      iters:",  np.round(i_ip_n[:,0], 1))

    # -----------------------------------------------
    # Experiment 2: Varying m (inequalities), fixed n
    # -----------------------------------------------
    n_fixed = [100]
    m_list  = [50, 100, 200, 500, 1000]

    print(f"\n[Experiment 2] Fixed n={n_fixed[0]}, varying m, tolerance={TOLERANCE}")
    t_simp_m, i_simp_m = run_experiment('highs-ds',  n_fixed, m_list, num_trials=5, tol=TOLERANCE)
    t_ip_m,   i_ip_m   = run_experiment('highs-ipm', n_fixed, m_list, num_trials=5, tol=TOLERANCE)

    plot_comparison(m_list, t_simp_m[0,:], t_ip_m[0,:],
                    "Number of Inequalities (m)", "Mean Computational Time (s)",
                    f"Computational Time vs m (fixed n={n_fixed[0]})",
                    "Time_vs_m")
    plot_comparison(m_list, i_simp_m[0,:], i_ip_m[0,:],
                    "Number of Inequalities (m)", "Mean Number of Iterations",
                    f"Iterations vs m (fixed n={n_fixed[0]})",
                    "Iters_vs_m")

    print("  m_list:", m_list)
    print("  Simplex times:", np.round(t_simp_m[0,:], 6))
    print("  IP      times:", np.round(t_ip_m[0,:], 6))
    print("  Simplex iters:", np.round(i_simp_m[0,:], 1))
    print("  IP      iters:", np.round(i_ip_m[0,:], 1))

    # -----------------------------------------------
    # Experiment 3: Sensitivity to Stopping Tolerance
    # -----------------------------------------------
    tol_list = [1e-4, 1e-6, 1e-8, 1e-10, 1e-12]
    n_tol, m_tol = 200, 200
    t_simp_tol, t_ip_tol = [], []
    i_simp_tol, i_ip_tol = [], []

    print(f"\n[Experiment 3] Tolerance sensitivity for n={n_tol}, m={m_tol}")
    for tol in tol_list:
        ts, is_ = run_experiment('highs-ds',  [n_tol], [m_tol], num_trials=3, tol=tol)
        ti, ii  = run_experiment('highs-ipm', [n_tol], [m_tol], num_trials=3, tol=tol)
        t_simp_tol.append(ts[0,0])
        t_ip_tol.append(ti[0,0])
        i_simp_tol.append(is_[0,0])
        i_ip_tol.append(ii[0,0])
        print(f"  tol={tol:.0e} | Simplex: t={ts[0,0]:.5f}s, iters={is_[0,0]:.0f}  |  IP: t={ti[0,0]:.5f}s, iters={ii[0,0]:.0f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    tol_labels = [f"{t:.0e}" for t in tol_list]

    ax1.plot(tol_labels, t_simp_tol, marker='o', linewidth=2, label='Simplex')
    ax1.plot(tol_labels, t_ip_tol, marker='s', linewidth=2, label='Interior-Point')
    ax1.set_title(f"Computational Time vs Tolerance (n={n_tol}, m={m_tol})", fontsize=13)
    ax1.set_xlabel("Stopping Tolerance", fontsize=12)
    ax1.set_ylabel("Mean Computational Time (s)", fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.7)

    ax2.plot(tol_labels, i_simp_tol, marker='o', linewidth=2, label='Simplex')
    ax2.plot(tol_labels, i_ip_tol, marker='s', linewidth=2, label='Interior-Point')
    ax2.set_title(f"Iterations vs Tolerance (n={n_tol}, m={m_tol})", fontsize=13)
    ax2.set_xlabel("Stopping Tolerance", fontsize=12)
    ax2.set_ylabel("Mean Number of Iterations", fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig("plots/Tolerance_Sensitivity.png", dpi=150)
    plt.close()

    print("\n" + "=" * 60)
    print("All experiments complete. Plots saved in 'plots/' directory.")
    print("=" * 60)


if __name__ == "__main__":
    main()
