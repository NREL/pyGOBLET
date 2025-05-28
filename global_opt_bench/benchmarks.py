import benchmark_functions as bf
import matplotlib.pyplot as plt
import numpy as np
import time
from inspect import signature

problems = bf.__2D__

## I need funcitons that do:
# run solvers - time
def run_solvers_time(solvers, problems=bf.__All__, test_dimensions= [2, 5, 10, 15], tol=1e-3, n_runs=10, verbose=False):
    """
    Run a list of solvers on a set of problems and return the time taken for each solver.
    Assumes that each solver takes as arguments a function to evaluate
    and either one or both of an initial point `x0` and bounds.
    Assumes that each solver returns an object with an attribute `x` that contains the solution.

    :param solvers: List of solver instances.
    :param problems: List of problem classes, defaults to ``bf.__All__``.
    :param test_dimensions: List of dimensions to test any n-dimensional problems on, defaults to ``[2, 5, 10, 15]``.
    :param tol: Tolerance of the solution. If :math:`x_{sol} - x_{opt} < tol`, the solution is considered successful, defaults to ``1e-3``.
    :param n_runs: Number of runs for each problem. Each solver will be run ``n_runs`` times on each problem, with different random seeds, defaults to ``10``.
    :param verbose: If True, prints progress of the run, defaults to ``False``.
    :return: List of dictionaries recording data in the form ``{'solver', 'problem', 'repeat', 'success', 'time'}``.
    """
    results = []
    for solver in solvers:
        for problem in problems:
            problem_dim = problem.DIM

            # Determine what dimensions to test
            if isinstance(problem_dim, tuple):
                # If problem has a range of dimensions, only test dimensions within the range
                if problem_dim[1] == -1:
                    dims_to_test = [d for d in test_dimensions if d >= problem_dim[0]]
                else:
                    dims_to_test = [d for d in test_dimensions if problem_dim[0] <= d < problem_dim[1]]
            else:
                dims_to_test = [problem_dim]

            for n_dims in dims_to_test:
                for i in range(n_runs):
                    # Ensures each problem is run with the same seed for each solver
                    np.random.seed(i)

                    # Initialize the problem instance for bounds access
                    problem_instance = problem(n_dims)

                    # Generate random initial point within bounds
                    bounds = np.array(problem_instance.bounds())
                    x0 = np.random.uniform(bounds[:, 0], bounds[:, 1])

                    if verbose:
                        print(f"Running {solver.__name__} on {problem} with dimensions {n_dims}, run {i+1}/{n_runs}")
                    
                    start_time = time.time()

                    # Run the solver
                    try:
                        solver_args = {}
                        sig = signature(solver)
                        if 'x0' in sig.parameters:
                            solver_args['x0'] = x0
                        if 'bounds' in sig.parameters:
                            solver_args['bounds'] = bounds
                        result = solver(problem_instance.evaluate, **solver_args)
                        point = result.x

                        # Check if solution is within tol of a global minimum
                        passed = np.any(np.linalg.norm(point - problem_instance.argmin(), axis=1) < tol)
                    except Exception as e:
                        point = None
                        passed = False

                    end_time = time.time()
                    results.append({
                        'solver': solver.__name__,
                        'problem': problem,
                        'repeat': i,
                        'success': passed,
                        # Record time taken as the metric
                        'metric': end_time - start_time
                    })
    return results

# run solvers - fxn evals

# compute performance ratios

# compute performance profiles

# plot performance profiles

## Dont really need any of the functions in here
