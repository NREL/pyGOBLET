import numpy as np
from pyomo.environ import SolverFactory
from pygold import get_standard_problems

# Shows how to use pyGOLD with Pyomo

# Choose benchmark problems to use
problems = get_standard_problems("All")

# Choose pyomo solver to use
# Here we use IPOPT, but you can choose any solver compatible with Pyomo
solver = SolverFactory('ipopt')

for problem in problems:
    print(f"\nOptimizing: {problem.__name__}")

    # If problem is n-dimensional, initialize it in 2D
    # Problems must have a specific dimension to be accessed as a pyomo model
    if isinstance(problem.DIM, tuple):
        initialized = problem(2)
    else:
        initialized = problem(problem.DIM)

    # Get the pyomo model
    model = initialized.as_pyomo_model()

    # Try to use the solver
    try:
        result = solver.solve(model, tee=False)

    # Some problems may not be compatible with the solver
    # due to issues such as non-differentiability
    except ValueError:
        print(f"Solver failed for {problem.__name__}. Solver may not be compatible with this problem.")
        continue

    # Take the optimal solution
    x_opt = np.array([model.x[i].value for i in range(initialized._ndims)])

    # Output the results
    # IPOPT will not find the global optimum for all problems
    # since it is a local solver
    print(f"Calculated Optimal x: {x_opt}")
    print(f"True Optimal x: {initialized.argmin()}")
