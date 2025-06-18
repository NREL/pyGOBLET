from collections import defaultdict
import os
import warnings
from inspect import signature
import matplotlib.pyplot as plt
import numpy as np
from pygold.postprocessing import testbed

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

def resolve_unknown_min_old(data, tol=1e-3):
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

def log_coco_from_results(results, output_folder="output_data"):
    """
    Write .dat, .tdat, and .info files in the COCOPP format from a list of
    solver/problem result dictionaries.

    :param results: List of dictionaries containing test information.
        Each dictionary should contain at least the keys:
        'solver', 'problem', 'func_id', 'n_dims',
        'log' (list of (fevals, gevals, fval)), 'min'
    :param output_folder: Directory to save the output files.
        Defaults to "output_data".
    """
    suite="pyGOLD"
    logger_name="bbob"
    data_format="bbob-new2"
    coco_version="2.7.3"
    precision=1e-8

    os.makedirs(output_folder, exist_ok=True)
    # Group by solver, func_id, n_dims
    grouped = defaultdict(list)
    for res in results:
        key = (res['solver'], res['func_id'], res['n_dims'], res['problem'])
        grouped[key].append(res)
    for (solver, func_id, n_dims, problem), runs in grouped.items():
        func_id += 1
        alg_folder = os.path.join(output_folder, solver)
        os.makedirs(alg_folder, exist_ok=True)
        dat_folder = f"data_{solver}_f{func_id}"
        dat_path = os.path.join(alg_folder, dat_folder)
        os.makedirs(dat_path, exist_ok=True)
        dat_file = os.path.join(dat_path, f"{solver}_f{func_id}_DIM{n_dims}.dat")
        tdat_file = os.path.join(dat_path, f"{solver}_f{func_id}_DIM{n_dims}.tdat")
        dat_rel_path = os.path.relpath(dat_file, alg_folder)
        evals_list = []
        fval_list = []
        min_val = None
        for res in runs:
            if 'min' in res and res['min'] is not None:
                min_val = res['min']
                break
        if min_val is None:
            warnings.warn(f"Minimum for {problem} in {n_dims}D is None, use smallest fval found as min and call this function again.")
            return
        with open(dat_file, 'w') as df, open(tdat_file, 'w') as tdf:
            for i, res in enumerate(runs):
                for f in [df, tdf]:
                    f.write(f"%% iter/random seed: {i+1}\n")
                    f.write(f"%% algorithm: {solver}\n")
                    f.write("%% columns: fevals gevals fval\n")
                for entry in res['log']:
                    if len(entry) == 3:
                        fevals, _, fval = entry
                    elif len(entry) == 2:
                        fevals, fval = entry
                    else:
                        continue
                    for f in [df, tdf]:
                        f.write(f"{fevals} 0 {fval - min_val}\n")
                evals_list.append(res.get('evals', len(res['log'])))
                last_fval = res['log'][-1][2] if len(res['log'][-1]) > 2 else res['log'][-1][1]
                fval_list.append(last_fval)
        info_file = os.path.join(alg_folder, f"{solver}_f{func_id}_DIM{n_dims}.info")
        dat_rel_path = os.path.relpath(tdat_file, alg_folder)
        info_header = f"suite = '{suite}', funcId = {func_id}, DIM = {n_dims}, Precision = {precision:.3e}, algId = '{solver}', logger = '{logger_name}', data_format = '{data_format}', coco_version = '{coco_version}'"
        info_comment = f"% Run {solver} on {problem} in {n_dims}D"
        info_data = f"{dat_rel_path}, " + ", ".join([f"{i+1}:{evals_list[i]}|{fval_list[i] - min_val}" for i in range(len(runs))])
        with open(info_file, 'w') as inf:
            inf.write(info_header + "\n")
            inf.write(info_comment + "\n")
            inf.write(info_data + "\n")

def configure_testbed(problems, test_dimensions=[2, 4, 5, 8, 10, 12], groups=None):
    """
    Configure a custom COCOPP testbed for benchmarking solvers on problems.
    This function sets up the testbed with the provided solvers and problems,
    allowing the use of the COCOPP post-processing framework.

    :param solvers: List of solver instances.
    :param problems: List of problem classes.
    :param test_dimensions: List of dimensions to test any n-dimensional
        problems on, defaults to ``[2, 4, 5, 8, 10, 12]``.
    :param groups: Optional dictionary mapping solver names to groups.
        If provided must be of the form of a ordered dict.
    """
    testbed.CustomTestbed.dims = test_dimensions
    testbed.CustomTestbed.dimensions_to_display = test_dimensions
    testbed.CustomTestbed.goto_dimension = test_dimensions[3]
    testbed.CustomTestbed.rldDimsOfInterest = [test_dimensions[2], test_dimensions[4]]
    testbed.CustomTestbed.tabDimsOfInterest = [test_dimensions[2], test_dimensions[4]]
    testbed.CustomTestbed.short_names = {id+1: problem.__name__ for id, problem in enumerate(problems)}
    testbed.CustomTestbed.nfxns = len(problems)
    testbed.CustomTestbed.functions_with_legend=(1, len(problems))
    testbed.CustomTestbed.last_function_number=len(problems)
    testbed.CustomTestbed.settings['dimensions_to_display'] = test_dimensions
    testbed.CustomTestbed.settings['goto_dimension'] = test_dimensions[3]
    testbed.CustomTestbed.settings['rldDimsOfInterest'] = [test_dimensions[2], test_dimensions[4]]
    testbed.CustomTestbed.settings['tabDimsOfInterest'] = [test_dimensions[2], test_dimensions[4]]
    testbed.CustomTestbed.settings['short_names'] = {id + 1: problem.__name__ for id, problem in enumerate(problems)}
    testbed.CustomTestbed.settings['functions_with_legend'] = (1, len(problems))
    testbed.CustomTestbed.settings['last_function_number'] = len(problems)

    if groups is not None:
        testbed.CustomTestbed.func_cons_groups = groups
