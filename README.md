# ME5414 Project - Linear Programming Solver Comparison

**Matriculation Number:** A0326854W
**Group:** B (Simplex Method vs Interior Point Method)

This repository contains the numerical experiments for the ME5414 Linear Programming project. 
The project compares the computational performance of the **Revised Simplex** algorithm and the **Interior-Point** method using test problems with varying sizes and tolerances.

## Implementation Details
The comparison is implemented generated in Python. 
Solvers used:
- Scipy's `linprog` with `method='revised simplex'`
- Scipy's `linprog` with `method='interior-point'`

Problem dimensions:
* $n$ represents the dimension of $x$.
* $m$ represents the number of inequalities $Ax \le b$.

## Structure
- `main.py`: The main script that runs the experiments and generates plots.
- `plots/`: Contains generated graphs for the performance evaluation.
- `test_lp.py`: Simple unit testing to ensure the random LP generation works correctly.

## Executing the code
Ensure you have `numpy`, `scipy`, and `matplotlib` installed:
```bash
pip install numpy scipy matplotlib
```
Run the main script:
```bash
python3 main.py
```
This will automatically generate the corresponding plots in the `plots/` directory showcasing the computational timings, iterations, and tolerance impacts.
