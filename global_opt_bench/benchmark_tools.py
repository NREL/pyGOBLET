from collections import defaultdict
from inspect import signature
import benchmark_functions as bf
import matplotlib.pyplot as plt
import numpy as np
import time

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
                        'n_dims': n_dims,
                        'success': passed,
                        # Record time taken as the metric
                        'metric': end_time - start_time
                    })
    return results

def count_calls(func):
    """Decorator that counts how many times a function is called."""
    def wrapper(*args, **kwargs):
        wrapper.calls += 1
        return func(*args, **kwargs)
    wrapper.calls = 0
    return wrapper

def run_solvers_fxn_evals(solvers, problems=bf.__All__, test_dimensions= [2, 5, 10, 15], tol=1e-3, n_runs=10, verbose=False):
    """
    Run a list of solvers on a set of problems and return the number of function evaluations taken by each solver.
    Assumes that each solver takes as arguments a function to evaluate
    and either one or both of an initial point `x0` and bounds.
    Assumes that each solver returns an object with an attribute `x` that contains the solution.

    :param solvers: List of solver instances.
    :param problems: List of problem classes, defaults to ``bf.__All__``.
    :param test_dimensions: List of dimensions to test any n-dimensional problems on, defaults to ``[2, 5, 10, 15]``.
    :param tol: Tolerance of the solution. If :math:`x_{sol} - x_{opt} < tol`, the solution is considered successful, defaults to ``1e-3``.
    :param n_runs: Number of runs for each problem. Each solver will be run ``n_runs`` times on each problem, with different random seeds, defaults to ``10``.
    :param verbose: If True, prints progress of the run, defaults to ``False``.
    :return: List of dictionaries recording data in the form ``{'solver', 'problem', 'repeat', 'success', 'numEvals'}``.
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

                    # Decorate the problem's evaluate function to count calls
                    problem_instance.evaluate = count_calls(problem_instance.evaluate)

                    # Generate random initial point within bounds
                    bounds = np.array(problem_instance.bounds())
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
                        result = solver(problem_instance.evaluate, **solver_args)
                        point = result.x

                        # Check if solution is within tol of a global minimum
                        passed = np.any(np.linalg.norm(point - problem_instance.argmin(), axis=1) < tol)
                    except Exception as e:
                        point = None
                        passed = False

                    results.append({
                        'solver': solver.__name__,
                        'problem': problem,
                        'repeat': i,
                        'n_dims': n_dims,
                        'success': passed,
                        # Record function calls as the metric
                        'metric': problem_instance.evaluate.calls
                    })
    return results

def compute_performance_ratios(results):
    """
    Compute performance ratios for each solver on each problem instance.
    The performance ratio for solver :math:`s` on problem :math:`p` is defined as:

        .. math::

            r_{p,s} =
            \\begin{cases}
                \\frac{t_{p,s}}{\\min\\{t_{p,s}: s \\in S\\}} & \\text{if convergence test passed,} \\\\
                \\infty & \\text{if convergence test failed.}
            \\end{cases}
    where :math:`t_{p,s}` is the test metric for solver :math:`s` on problem :math:`p`, and :math:`S` is the set of all solvers.

    :param results: List of dictionaries, each with keys:
        'solver': str, name of the solver,
        'problem': class, the problem class,
        'repeat': int, repeat/run index,
        'n_dims': int, number of dimensions,
        'success': bool, whether the solver succeeded,
        'metric': float, value of the metric (e.g., time taken).
    :return: Nested dictionary of the form {(problem, n_dims, repeat): {solver: ratio, ...}, ...}.
    """
    # Group results by problem 
    # Each combination of (problem, n_dims, iteration) is considered a unique test
    grouped = defaultdict(dict)
    for res in results:
        key = (res['problem'], res['n_dims'], res['repeat'])
        grouped[key][res['solver']] = res
    
    ratios = defaultdict(dict)

    for key, solver_dict in grouped.items():
        # Get the minimum metric
        min_metric = min(solver_dict[solver]['metric'] for solver in solver_dict)

        for solver, res in solver_dict.items():
            # Calculate the performance ratio for each solver
            if res['success']:
                if min_metric == 0:
                    ratios[key][solver] = float('inf')
                else:
                    ratios[key][solver] = res['metric'] / min_metric
            else:
                ratios[key][solver] = float('inf')
        
    return ratios

def compute_performance_profiles(ratios, tau_grid=np.linspace(1, 10, 1500)):
    """
    Compute the performance profiles of each solver.
    The performance profile for solver :math:`s` is defined as:

        .. math::

            \\rho_{s}(\\tau) = \\frac{1}{|P|}\\text{size}\\{ p \\in P: r_{p,s} \\leq \\tau \\}

    where :math:`r_{p,s}` is the performance ratio for solver :math:`s` on problem :math:`p`, and :math:`|P|` is the total number of solvers.

    :param ratios: Nested dictionary of the form {(problem, n_dims, repeat): {solver: ratio, ...}, ...}.
    :param tau_grid: Optional list values to evaluate the performance profile at. Defaults to 100 points linearly spaced between 1 and 10.
    :return: Dictionary of the form {solver: (tau_grid, rho_values)}
    """
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

    :param profiles: Dictionary {solver: (tau_grid, rho_values)} or list of such dicts.
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
