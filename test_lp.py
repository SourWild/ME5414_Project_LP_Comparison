"""
Unit tests for ME5414 LP Solver Comparison Project
Matriculation No.: A0326854W
"""
import numpy as np
from scipy.optimize import linprog
from main import generate_random_lp, solve_lp


def test_generation_shapes():
    """Test that generate_random_lp returns correctly shaped arrays."""
    n, m = 10, 5
    c, A, b = generate_random_lp(n, m, seed=42)
    # A includes m original constraints + n upper-bound constraints
    assert len(c) == n,       f"Expected c of length {n}, got {len(c)}"
    assert A.shape == (m + n, n), f"Expected A shape ({m+n},{n}), got {A.shape}"
    assert len(b) == m + n,   f"Expected b of length {m+n}, got {len(b)}"
    print("PASS: test_generation_shapes")


def test_feasibility():
    """Test that generated problems are feasible and bounded."""
    for seed in range(10):
        n, m = 20, 15
        c, A, b = generate_random_lp(n, m, seed=seed)
        res = linprog(c, A_ub=A, b_ub=b, method='highs', bounds=(0, None))
        assert res.success, f"Problem with seed={seed} is infeasible or unbounded: {res.message}"
    print("PASS: test_feasibility (10 random problems)")


def test_solve_lp_simplex():
    """Test solve_lp returns valid result with Simplex method."""
    n, m = 30, 20
    c, A, b = generate_random_lp(n, m, seed=0)
    success, elapsed, n_iter = solve_lp(c, A, b, method='highs-ds', tol=1e-8)
    assert success, "Simplex solver failed on test problem"
    assert elapsed >= 0, "Negative elapsed time"
    assert n_iter >= 0, "Negative iteration count"
    print(f"PASS: test_solve_lp_simplex (t={elapsed:.5f}s, iters={n_iter})")


def test_solve_lp_interior_point():
    """Test solve_lp returns valid result with Interior-Point method."""
    n, m = 30, 20
    c, A, b = generate_random_lp(n, m, seed=0)
    success, elapsed, n_iter = solve_lp(c, A, b, method='highs-ipm', tol=1e-8)
    assert success, "Interior-Point solver failed on test problem"
    assert elapsed >= 0, "Negative elapsed time"
    assert n_iter >= 0, "Negative iteration count"
    print(f"PASS: test_solve_lp_interior_point (t={elapsed:.5f}s, iters={n_iter})")


def test_same_optimal_value():
    """
    Test that Simplex and Interior-Point methods produce the same optimal value
    on a known small LP problem.
    """
    # Simple LP: min c^T x s.t. Ax <= b, x >= 0
    # Known optimal: x* = [0, 1], optimal value = -1
    c = np.array([1.0, -1.0])
    A = np.array([[1.0, 1.0],
                  [1.0, 0.0],
                  [0.0, 1.0]])
    b = np.array([2.0, 1.0, 1.0])

    res_simp = linprog(c, A_ub=A, b_ub=b, method='highs-ds', bounds=(0, None))
    res_ip   = linprog(c, A_ub=A, b_ub=b, method='highs-ipm', bounds=(0, None))

    assert res_simp.success, "Simplex failed on known LP"
    assert res_ip.success,   "Interior-Point failed on known LP"
    assert abs(res_simp.fun - (-1.0)) < 1e-6, f"Simplex wrong optimal value: {res_simp.fun}"
    assert abs(res_ip.fun - (-1.0)) < 1e-6,   f"IP wrong optimal value: {res_ip.fun}"
    assert abs(res_simp.fun - res_ip.fun) < 1e-6, "Simplex and IP disagree on optimal value"
    print(f"PASS: test_same_optimal_value (Simplex={res_simp.fun:.6f}, IP={res_ip.fun:.6f})")


if __name__ == "__main__":
    print("=" * 50)
    print("Running Unit Tests for ME5414 LP Project")
    print("=" * 50)
    test_generation_shapes()
    test_feasibility()
    test_solve_lp_simplex()
    test_solve_lp_interior_point()
    test_same_optimal_value()
    print("=" * 50)
    print("All tests PASSED.")
    print("=" * 50)
