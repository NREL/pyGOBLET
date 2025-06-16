import numpy as np
from pyomo.environ import SolverFactory
import pygold.benchmark_functions as bf

# Runs pyomo local optimizer on all benchmark problems
# Runs on both constrained and unconstrained problems

# List of benchmark problems to test
problems = bf.__All__
print(problems)

# Note that ipopt is a local solver - not global
solver = SolverFactory('ipopt')

for problem in problems:
    print(f"\nOptimizing: {problem.__name__}")
    if isinstance(problem.DIM, tuple):
        initialized = problem(2)
    else:
        initialized = problem(problem.DIM)

    model = initialized.as_pyomo_model()
    try:
        result = solver.solve(model, tee=False)
    except ValueError:
        print(f"Solver failed for {problem.__name__}. Solver may not be compatible with this problem.")
        continue

    x_opt = np.array([model.x[i].value for i in range(initialized._ndims)])

    print(f"Calculated Optimal x: {x_opt}")
    print(f"True Optimal x: {initialized.argmin()}")
