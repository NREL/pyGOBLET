import numpy as np
from pyomo.environ import SolverFactory
import benchmark_functions as bf

# List of benchmark problems to test (choose a few for demonstration)
problems = [
    bf.Ackley(2),
    bf.Bukin6(2),
    bf.CrossInTray(2),
    bf.DropWave(2),
    bf.EggHolder(2)
]

# Note that ipopt is a local solver - not global
solver = SolverFactory('ipopt')

for problem in problems:
    print(f"\nOptimizing: {problem.__class__.__name__}")

    model = problem.as_pyomo_model()
    try:
        result = solver.solve(model, tee=False)
    except ValueError:
        print(f"Solver failed for {problem.__class__.__name__}. Solver may not be compatible with this problem.")
        continue

    x_opt = np.array([model.x[i].value for i in range(problem._ndims)])

    print(f"Calculated Optimal x: {x_opt}")
    print(f"True Optimal x: {problem.argmin()}")
