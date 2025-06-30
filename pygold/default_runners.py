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

def run_standard(solvers, problems, test_dimensions=[2, 4, 5, 8, 10, 12], n_iters=5, normalize=True, verbose=False):
    """
    Run a list of solvers on a set of problems from pyGOLD's standard problems
    module and generate log files in the COCO format.

    Each solver is assumed to take as arguments a function to evaluate and
    some combination of an initial point `x0`, bounds, and constraints.

    Data is recorded to output_data/ in the COCO format, which includes the
    number of function evaluations and the difference between the solution and
    the best known minimum. If the true function minimum is unknown, the
    smallest calculated function value is used as the best known minimum.

    :param solvers: List of solver instances.
    :param problems: List of problem classes from the standard problems module.
    :param test_dimensions: List of dimensions to test any n-dimensional
        problems on, defaults to ``[2, 4, 6, 8, 10, 12]``.
    :param n_iters: Number of runs for each problem. Each solver will be run
        ``n_iters`` times on each problem, with different random seeds
        per run consistent across solvers, defaults to ``5``.
    :param normalize: If True, normalize the fval - fmin value by dividing by
        the observed range of the function values. This allows for fair
        comparison between problems with significantly different scales.
        If False, the raw fval - fmin values are used.
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
                            solver_args['constraints'] = [{'type': 'ineq', 'fun': lambda x: constraint(x)} for constraint in prob.constraints()]
                        if 'minimizer_kwargs' in sig.parameters:
                            solver_args['minimizer_kwargs'] = {'constraints': [{'type': 'ineq', 'fun': lambda x: constraint(x)} for constraint in prob.constraints()]}
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

            # Find max val if normalizing
            if normalize:
                max_val = np.max([e[1] for res in results for e in res['log']])
                for res in results:
                    res['max'] = max_val

            # Save results to file in COCO format
            log_coco_from_results(results, normalize=normalize)

def run_floris(solvers, problems, n_turbines=[2, 4, 5, 8, 10, 12], n_iters=5, normalize=True, verbose=False):
    """
    Run a list of solvers on a set of problems from the FLORIS module
    and generate log files in the COCO format.

    Each solver is assumed to take as arguments a function to evaluate and
    some combination of an initial point `x0`, bounds, and constraints.

    Data is recorded to output_data/ in the COCO format, which includes the
    number of function evaluations and the difference between the solution and
    the best known minimum. If the true function minimum is unknown, the
    smallest calculated function value is used as the best known minimum.

    :param solvers: List of solver instances.
    :param problems: List of problem classes from the FLORIS problems module.
    :param n_turbines: List of turbine counts to test, defaults to
        ``[2, 4, 5, 8, 10, 12]``.
    :param n_iters: Number of runs for each problem. Each solver will be run
        ``n_iters`` times on each problem, with different random seeds
        per run consistent across solvers, defaults to ``5``.
    :param normalize: If True, normalize the fval - fmin value by dividing by
        the observed range of the function values. This allows for fair
        comparison between problems with significantly different scales.
        If False, the raw fval - fmin values are used.
    :param verbose: If True, prints progress of the run, defaults to ``False``.
    """

    for id, problem in enumerate(problems):

        for n_turb in n_turbines:
            results = []
            prob = problem(n_turb)

            orig_eval = prob.evaluate

            for solver in solvers:
                for i in range(n_iters):
                    np.random.seed(i)  # Ensure reproducibility between solvers

                    # Invert the objective to make it a minimization problem
                    # And wrap the problem with a logger
                    prob.evaluate = logger(lambda *args, **kwargs: -orig_eval(*args, **kwargs))

                    # Generate initial layout
                    attempts = 0
                    while attempts < 1000000:
                        # Generate random layout
                        x = np.random.uniform(0, 1000, size=(n_turb, 2))

                        # Sort turbines to satisfy permutation constraint
                        x = x[np.argsort(x[:, 0])]

                        # Check constraints
                        dist_constraints = prob.dist_constraint(x)
                        perm_constraints = prob.perm_constraint(x)
                        if np.all(dist_constraints >= 0) and np.all(perm_constraints >= 0):
                            break
                        attempts += 1

                    if attempts == 1000000:
                        warnings.warn(f"Failed to generate valid initial layout for {problem.__name__} with {n_turb} turbines after 1000000 attempts. Skipping solver run.")
                        continue

                    if prob.DIM == 3:
                        # Add random yaw angles close to zero
                        x = np.hstack((x, np.random.uniform(-np.pi/32, np.pi/32, size=(n_turb, 1))))

                    if verbose:
                        print(f"Running {solver.__name__} on {problem.__name__} with {n_turb} turbines, {n_turb * problem.DIM} dimensions, iteration {i+1}/{n_iters}")

                    # Run the solver
                    try:
                        solver_args = {}
                        sig = signature(solver)
                        if 'x0' in sig.parameters:
                            solver_args['x0'] = x.flatten()
                        if 'bounds' in sig.parameters:
                            solver_args['bounds'] = prob.bounds().reshape(n_turb * prob.DIM, 2)
                        if 'constraints' in sig.parameters:
                            solver_args['constraints'] = [{'type': 'ineq', 'fun': lambda x: constraint(x)} for constraint in prob.constraints()]
                        if 'minimizer_kwargs' in sig.parameters:
                            solver_args['minimizer_kwargs'] = {'constraints': [{'type': 'ineq', 'fun': lambda x: constraint(x)} for constraint in prob.constraints()]}
                        _ = solver(prob.evaluate, **solver_args)
                    except Exception as e:
                        warnings.warn(f"Solver {solver.__name__} failed on {problem} with {n_turb} turbines, {n_turb * problem.DIM} dimensions, run {i+1}/{n_iters}: {e}")
                        continue

                    results.append({'solver': solver.__name__,
                                    'problem': problem.__name__,
                                    'func_id': id,
                                    'random_seed': i,
                                    'n_dims': n_turb * problem.DIM,
                                    'min': None,
                                    'log': prob.evaluate.log,
                                    })
            # Results for this problem and dimension are now complete
            # Resolve unknown min case
            results = resolve_unknown_min(results)

            # Find max val if normalizing
            if normalize:
                max_val = np.max([e[1] for res in results for e in res['log']])
                for res in results:
                    res['max'] = max_val

            # Save results to file in COCO format
            log_coco_from_results(results, normalize=normalize)

def run_solvers(solvers, problems, test_dimensions=[2, 4, 5, 8, 10, 12], n_iters=5, normalize=True, verbose=False):
    """
    Run a list of solvers on a set of problems and generate log files in the
    COCO format.

    Each solver is assumed to take as arguments a function to evaluate and
    some combination of an initial point `x0`, bounds, and constraints.

    Problems can be from the standard problems module or the Floris module.
    Floris problems use test_dimensions to specify the number of turbines
    to test. If a problem is not from either module, the standard problem
    runner is used.

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
    :param normalize: If True, normalize the fval - fmin value by dividing by
        the observed range of the function values. This allows for fair
        comparison between problems with significantly different scales.
        If False, the raw fval - fmin values are used.
    :param verbose: If True, prints progress of the run, defaults to ``False``.
    """

    standard_problems = []
    floris_problems = []
    for p in problems:
        if p.__module__.startswith('pygold.problems.standard'):
            standard_problems.append(p)
        elif p.__module__.startswith('pygold.problems.floris'):
            floris_problems.append(p)
        else:
            warnings.warn(f"Problem {p.__name__} is not a standard or FLORIS problem, trying to use the standard problem runner.")

    run_standard(solvers, standard_problems, test_dimensions=test_dimensions, n_iters=n_iters, normalize=normalize, verbose=verbose)

    run_floris(solvers, floris_problems, n_turbines=test_dimensions, n_iters=n_iters, normalize=normalize, verbose=verbose)
