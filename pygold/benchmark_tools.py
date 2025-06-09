from collections import defaultdict
from inspect import signature
import matplotlib.pyplot as plt
import numpy as np
import time

def run_solvers_time(solvers, problems, test_dimensions= [2, 5, 10, 15], tol=1e-3, n_runs=10, verbose=False):
    """
    Run a list of solvers on a set of problems and return the time taken for
    each solver. Assumes that each solver takes as arguments a function to
    evaluate and either one or both of an initial point `x0` and bounds.
    Assumes that each solver returns an object with an attribute
    `x` that contains the solution.

    :param solvers: List of solver instances.
    :param problems: List of problem classes.
    :param test_dimensions: List of dimensions to test any n-dimensional
        problems on, defaults to ``[2, 5, 10, 15]``.
    :param tol: Tolerance of the solution. If :math:`x_{sol} - x_{opt} < tol`,
        the solution is considered successful, defaults to ``1e-3``.
    :param n_runs: Number of runs for each problem.
        times on each problem, with different random seeds, defaults to ``10``.
    :param verbose: If True, prints progress of the run, defaults to ``False``.
    :return: List of dictionaries recording data in the form
        ``{'solver', 'problem', 'random_seed', 'n_dims', 'point', 'success',
        'metric'}``.
    """
    results = []
    for solver in solvers:
        for problem in problems:
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
                for i in range(n_runs):
                    # Ensures each problem is run
                    # with the same seed for each solver
                    np.random.seed(i)

                    # Initialize the problem instance for bounds access
                    problem_instance = problem(n_dims)

                    # Generate random initial point within bounds
                    bounds = np.array(problem_instance.bounds())
                    bounds[np.isneginf(bounds)] = -2000
                    bounds[np.isposinf(bounds)] = 2000
                    x0 = np.random.uniform(bounds[:, 0], bounds[:, 1])
                    if len(problem_instance.constraints()) > 0:
                        while not np.all([constraint(np.array(x0)) >= 0 for constraint in problem_instance.constraints()]):
                            x0 = np.random.uniform(bounds[:, 0], bounds[:, 1])

                    if verbose:
                        print(f"Running {solver.__name__} on {problem} with dimensions {n_dims}, run {i+1}/{n_runs}")

                    start_time = time.perf_counter()

                    # Run the solver
                    try:
                        solver_args = {}
                        sig = signature(solver)
                        if 'x0' in sig.parameters:
                            solver_args['x0'] = x0
                        if 'bounds' in sig.parameters:
                            solver_args['bounds'] = bounds
                        if 'constraints' in sig.parameters:
                            solver_args['constraints'] = [{'type': 'eq', 'fun': lambda x: constraint(x)} for constraint in problem_instance.constraints()]
                        if 'minimizer_kwargs' in sig.parameters:
                            solver_args['minimizer_kwargs'] = {'constraints': [{'type': 'eq', 'fun': lambda x: constraint(x)} for constraint in problem_instance.constraints()]}
                        result = solver(problem_instance.evaluate, **solver_args)
                        point = result.x

                        # Check if solution is within tol of a global minimum
                        if problem_instance.argmin() is not None:
                            # If the problem has a known minimum
                            # Check if the solution is within tolerance
                            passed = np.any(np.linalg.norm(point - problem_instance.argmin(), axis=1) < tol)
                        else:
                            passed = None
                    except Exception:
                        point = None
                        passed = False

                    end_time = time.perf_counter()
                    results.append({
                        'solver': solver.__name__,
                        'problem': problem,
                        'random_seed': i,
                        'n_dims': n_dims,
                        'point': point,
                        'success': passed,
                        # Record time taken as the metric
                        'metric': end_time - start_time
                    })
    return results

def count_calls(func):
    """
    Decorator that counts the number of times a function is called.

    When applied to a function, this decorator adds a ``calls`` attribute
    which increments each time the function is invoked.

    :returns: Wrapped version of the original function
        with a ``calls`` attribute.
    """
    def wrapper(*args, **kwargs):
        wrapper.calls += 1
        return func(*args, **kwargs)
    wrapper.calls = 0
    return wrapper

def run_solvers_fxn_evals(solvers, problems, test_dimensions= [2, 5, 10, 15], tol=1e-3, n_runs=10, verbose=False):
    """
    Run a list of solvers on a set of problems and return the number of function
    evaluations taken by each solver.
    Assumes that each solver takes as arguments a function to evaluate
    and either one or both of an initial point `x0` and bounds.
    Assumes that each solver returns an object with an attribute `x`
    that contains the solution.

    :param solvers: List of solver instances.
    :param problems: List of problem classes.
    :param test_dimensions: List of dimensions to test any n-dimensional
        problems on, defaults to ``[2, 5, 10, 15]``.
    :param tol: Tolerance of the solution. If :math:`x_{sol} - x_{opt} < tol`,
        the solution is considered successful, defaults to ``1e-3``.
    :param n_runs: Number of runs for each problem. Each solver will be run
        ``n_runs`` times on each problem, with different random seeds,
        defaults to ``10``.
    :param verbose: If True, prints progress of the run, defaults to ``False``.
    :return: List of dictionaries recording data in the form
        ``{'solver', 'problem', 'random_seed', 'n_dims', 'point', 'success',
        'metric'}``.
    """
    results = []
    for solver in solvers:
        for problem in problems:
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
                for i in range(n_runs):
                    # Ensures each problem is run
                    # with the same seed for each solver
                    np.random.seed(i)

                    # Initialize the problem instance for bounds access
                    problem_instance = problem(n_dims)

                    # Decorate the problem's evaluate function to count calls
                    problem_instance.evaluate = count_calls(problem_instance.evaluate)

                    # Generate random initial point within bounds
                    bounds = np.array(problem_instance.bounds())
                    bounds[np.isneginf(bounds)] = -2000
                    bounds[np.isposinf(bounds)] = 2000
                    x0 = np.random.uniform(bounds[:, 0], bounds[:, 1])
                    if len(problem_instance.constraints()) > 0:
                        while not np.all([constraint(x0) >= 0 for constraint in problem_instance.constraints()]):
                            x0 = np.random.uniform(bounds[:, 0], bounds[:, 1])

                    if verbose:
                        print(f"Running {solver.__name__} on {problem} with dimensions {n_dims}, run {i+1}/{n_runs}")


                    # Run the solver
                    try:
                        solver_args = {}
                        sig = signature(solver)
                        if 'x0' in sig.parameters:
                            solver_args['x0'] = x0
                        if 'bounds' in sig.parameters:
                            solver_args['bounds'] = bounds
                        if 'constraints' in sig.parameters:
                            solver_args['constraints'] = [{'type': 'eq', 'fun': lambda x: constraint(x)} for constraint in problem_instance.constraints()]
                        if 'minimizer_kwargs' in sig.parameters:
                            solver_args['minimizer_kwargs'] = {'constraints': [{'type': 'eq', 'fun': lambda x: constraint(x)} for constraint in problem_instance.constraints()]}
                        result = solver(problem_instance.evaluate, **solver_args)
                        point = result.x

                        if problem_instance.argmin() is not None:
                            # If the problem has a known minimum
                            # Check if the solution is within tolerance
                            passed = np.any(np.linalg.norm(point - problem_instance.argmin(), axis=1) < tol)
                        else:
                            passed = None
                    except Exception:
                        point = None
                        passed = False

                    results.append({
                        'solver': solver.__name__,
                        'problem': problem,
                        'random_seed': i,
                        'n_dims': n_dims,
                        'point': point,
                        'success': passed,
                        # Record function calls as the metric
                        'metric': problem_instance.evaluate.calls
                    })
    return results

def resolve_unknown_min(data, tol=1e-3):
    """
    When the minimum of a problem is unknown,
    'success' is set to None as there is no known minimum to compare against.

    This function updates the results to use the minimum found point
    as the 'argmin' for those problems where the minimum is unknown and
    uses this point to determine success.

    :param data: List of dictionaries containing test information.
        Each dictionary has the following keys:
        - ``solver``: Name of the solver.
        - ``problem``: The problem class being tested.
        - ``random_seed``: The random seed used for the test.
        - ``n_dims``: Number of dimensions for the problem.
        - ``point``: The minimum point found by the solver.
        - ``success``: Whether the solver successfully solved the problem.
        - ``metric``: Value of the evaluation metric (e.g., time taken).

    :return: List of dictionaries recording data in the form
        ``{'solver', 'problem', 'random_seed', 'n_dims', 'point', 'success',
        'metric'}``.
    """
    for res in data:
        if res['success'] is None:
            best = min([res['problem'].evaluate(r['point']) for r in data if r['problem'] == res['problem'] and r['point'] is not None and r['n_dims'] == res['n_dims']])
            res['success'] = abs(res['problem'].evaluate(res['point']) - best) < tol
    return data

def compute_performance_ratios(data):
    """
    Compute performance ratios for each solver on each problem instance.
    The performance ratio for solver :math:`s` on problem :math:`p`
    is defined as:

        .. math::

            r_{p,s} =
            \\begin{cases}
                \\frac{t_{p,s}}{\\min\\{t_{p,s}: s \\in S\\}}
                & \\text{if convergence test passed,} \\\\
                \\infty & \\text{if convergence test failed.}
            \\end{cases}

    where :math:`t_{p,s}` is the test metric for solver :math:`s` on problem
    :math:`p`, and :math:`S` is the set of all solvers.

    :param data: A list of dictionaries containing test information.
        Each dictionary has the following keys:
    - ``solver`` (*str*): Name of the solver.
    - ``problem`` (*class*): The problem class being tested.
    - ``random_seed`` (*int*): The random seed used for the test.
    - ``n_dims`` (*int*): Number of dimensions for the problem.
    - ``success`` (*bool*): Whether the solver successfully solved the problem.
    - ``metric`` (*float*): Value of the evaluation metric (e.g., time taken).

    :return: Nested dictionary of the form
        ``{(problem, n_dims, random_seed): {solver: ratio, ...}, ...}``.
    """
    # Group results by problem
    # Each combination of (problem, n_dims, iteration)
    # is considered a unique test
    grouped = defaultdict(dict)
    for res in data:
        key = (res['problem'], res['n_dims'], res['random_seed'])
        grouped[key][res['solver']] = res

    ratios = defaultdict(dict)

    for key, solver_dict in grouped.items():
        # Get the minimum metric
        try:
            min_metric = min(solver_dict[solver]['metric'] for solver in solver_dict if solver_dict[solver]['success'])

            for solver, res in solver_dict.items():
                # Calculate the performance ratio for each solver
                if res['success']:
                    if min_metric == 0:
                        ratios[key][solver] = float('inf')
                    else:
                        ratios[key][solver] = res['metric'] / min_metric
                else:
                    ratios[key][solver] = float('inf')
        except ValueError:
            # If no solver succeeded, all ratios are infinite
            for solver in solver_dict:
                ratios[key][solver] = float('inf')

    return ratios

def compute_performance_profiles(ratios, tau_grid=None):
    """
    Compute the performance profiles of each solver.
    The performance profile for solver :math:`s` is defined as:

        .. math::

            \\rho_{s}(\\tau) = \\frac{1}{|P|}\\text{size}\\{ p \\in P: r_{p,s}
            \\leq \\tau \\}

    where :math:`r_{p,s}` is the performance ratio for solver :math:`s` on
    problem :math:`p`, and :math:`|P|` is the total number of problems.

    :param ratios: Nested dictionary of the form
        ``{(problem, n_dims, random_seed): {solver: ratio, ...}, ...}``.
    :param tau_grid: Optional list values to evaluate the performance
        profile at. Defaults to 100 points linearly spaced between 1 and 10.
    :return: Dictionary of the form ``{solver: (tau_grid, rho_values)}``.
    """
    if tau_grid is None:
        tau_grid = np.linspace(1, 10, 1500)

    profiles = {}
    n_problems = len(ratios)
    if n_problems == 0:
        return profiles
    solvers = list(next(iter(ratios.values())).keys())

    for solver in solvers:
        # Gather all ratios for the current solver
        r_s = [ratios[key][solver] for key in ratios]
        # Calculate the performance profile for the solver
        rho = [(np.sum(np.array(r_s) <= tau) / n_problems) for tau in tau_grid]
        profiles[solver] = (tau_grid, rho)

    return profiles

def plot_performance_profiles(profiles, metrics=None):
    """
    Plots the performance profiles for each solver.

    :param profiles: Dictionary or list of dictionaries of the form
        ``{solver: (tau_grid, rho_values)}``.
    :param metrics: Optional string or list of strings for subplot titles.
    """
    if isinstance(profiles, list):
        n = len(profiles)
        fig, axes = plt.subplots(1, n, figsize=(8 * n, 6), squeeze=False)
        for i, prof in enumerate(profiles):
            ax = axes[0, i]
            for solver, (tau, rho) in prof.items():
                ax.plot(tau, rho, label=solver)
            ax.set_xlabel(r'$\tau$')
            ax.set_ylabel(r'Profile $\rho_s(\tau)$')
            title = 'Performance Profiles'
            if metrics is not None:
                if isinstance(metrics, list):
                    if i < len(metrics):
                        title += f' ({metrics[i]})'
                else:
                    title += f' ({metrics})'
            ax.set_title(title)
            ax.legend()
            ax.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        plt.figure(figsize=(8, 6))
        for solver, (tau, rho) in profiles.items():
            plt.plot(tau, rho, label=solver)
        plt.xlabel(r'$\tau$')
        plt.ylabel(r'Profile $\rho_s(\tau)$')
        title = 'Performance Profiles'
        if metrics is not None:
            title += f' ({metrics})'
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
