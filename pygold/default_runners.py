import warnings
from inspect import signature
from pygold.cocopp_interface.interface import log_coco_from_results
import numpy as np

def logger(func):
    """
    Decorator to record the function calls and function values
    for a given function. Adds a `log` attribute to the function
    that stores a list of tuples, where each tuple contains the
    number of calls and the result of the function call.

    :param func: The function to be decorated.
    :return: The wrapped function.
    """
    def wrapper(*args, **kwargs):
        wrapper.calls += 1
        result = func(*args, **kwargs)
        wrapper.log.append((wrapper.calls, result))
        return result
    wrapper.calls = 0
    wrapper.log = []
    return wrapper

def run_solvers(solvers, problems, test_dimensions=[2, 4, 5, 8, 10, 12], n_iters=5, verbose=False):
    """
    Run a list of solvers on a set of problems and generate log files in the
    COCO format.

    Each solver is assumed to take as arguments a function to evaluate and
    some combination of an initial point `x0`, bounds, and constraints.

    Data is recorded to output_data/ in the COCO format, which includes the
    number of function evaluations and the difference between the solution and
    the best known minimum. If the true function minimum is unknown, the
    smallest calculated function value is used as the best known minimum.

    :param solvers: List of solver instances.
    :param problems: List of problem classes.
    :param test_dimensions: List of dimensions to test any n-dimensional
        problems on, defaults to ``[2, 4, 6, 8, 10, 12]``.
    :param n_iters: Number of runs for each problem. Each solver will be run
        ``n_iters`` times on each problem, with different random seeds
        per run consistent across solvers, defaults to ``5``.
    :param verbose: If True, prints progress of the run, defaults to ``False``.
    """

    for id, problem in enumerate(problems):
        problem_dim = problem.DIM

        # Determine what dimensions to test
        if isinstance(problem_dim, tuple):
            # If problem has a range of dimensions,
            # only test dimensions within the range
            if problem_dim[1] == -1:
                dims_to_test = [d for d in test_dimensions if d >= problem_dim[0]]
            else:
                dims_to_test = [d for d in test_dimensions if problem_dim[0] <= d < problem_dim[1]]
        else:
            dims_to_test = [problem_dim]

        for n_dims in dims_to_test:
            results = []
            prob = problem(n_dims)

            for solver in solvers:
                for i in range(n_iters):
                    np.random.seed(i)  # Ensure reproducibility between solvers

                    # Wrap the problem with a logger
                    prob.evaluate = logger(prob.evaluate)

                    # Generate initial point within bounds
                    bounds = np.array(prob.bounds())
                    bounds[np.isneginf(bounds)] = -2000
                    bounds[np.isposinf(bounds)] = 2000
                    x0 = np.random.uniform(bounds[:, 0], bounds[:, 1])
                    if len(prob.constraints()) > 0:
                        while not np.all([constraint(x0) >= 0 for constraint in prob.constraints()]):
                            x0 = np.random.uniform(bounds[:, 0], bounds[:, 1])

                    if verbose:
                        print(f"Running {solver.__name__} on {problem.__name__} in {n_dims}D, iteration {i+1}/{n_iters}")

                    # Run the solver
                    try:
                        solver_args = {}
                        sig = signature(solver)
                        if 'x0' in sig.parameters:
                            solver_args['x0'] = x0
                        if 'bounds' in sig.parameters:
                            solver_args['bounds'] = bounds
                        if 'constraints' in sig.parameters:
                            solver_args['constraints'] = [{'type': 'eq', 'fun': lambda x: constraint(x)} for constraint in prob.constraints()]
                        if 'minimizer_kwargs' in sig.parameters:
                            solver_args['minimizer_kwargs'] = {'constraints': [{'type': 'eq', 'fun': lambda x: constraint(x)} for constraint in prob.constraints()]}
                        _ = solver(prob.evaluate, **solver_args)
                    except Exception as e:
                        warnings.warn(f"Solver {solver.__name__} failed on {problem} with dimensions {n_dims}, run {i+1}/{n_iters}: {e}")
                        continue

                    results.append({'solver': solver.__name__,
                                    'problem': problem.__name__,
                                    'func_id': id,
                                    'random_seed': i,
                                    'n_dims': n_dims,
                                    'min': prob.min(),
                                    'log': prob.evaluate.log,
                                    })
            # Results for this problem and dimension are now complete
            # Resolve unknown min case
            results = resolve_unknown_min(results)

            # Save results to file in COCO format
            log_coco_from_results(results)

def resolve_unknown_min(data):
    """
    When the minimum of a problem is unknown, use the
    minimum found function value as the min.

    :param data: List of dictionaries containing test information.
        Each dictionary must have the following keys:
        - ``min``: The minimum value function value, or None if unknown.
        - ``log``: Solver log in form [(fcalls, fvals)].
        - ``problem``: The problem class being tested.
        - ``n_dims``: Number of dimensions for the problem.

    :return: List of dictionaries with updated 'min' values.
        Each dictionary will have the 'min' key updated to the minimum function
        value found in the logs if it was initially None.
    """
    for res in data:
        if res['min'] is None:
            # Collect all logs for the same problem and n_dims
            logs = [r['log'] for r in data if r['problem'] == res['problem'] and r['n_dims'] == res['n_dims']]

            # Find the minimum function value from the logs
            min_value = min([min(sublist, key=lambda x: x[1])[1] for sublist in logs])
            res['min'] = min_value
    return data
