# Copyright (c) 2025 Alliance for Sustainable Energy, LLC

import warnings
import os
import numpy as np
import pandas as pd
from inspect import signature
from pygold.cocopp_interface.interface import log_coco_from_results
from pygold.optimizer import BaseOptimizer
try:
    from codecarbon import EmissionsTracker
    CODECARBON_AVAILABLE = True
except ImportError:
    CODECARBON_AVAILABLE = False
    EmissionsTracker = None

def logger(func, bounds=None):
    """
    Decorator to record the function calls and function values
    for a given function. Adds a `log` attribute to the function
    that stores a list of tuples, where each tuple contains the
    number of calls and the result of the function call.
    Optionally checks if the input arguments are within given bounds.

    :param func: The function to be decorated.
    :param bounds: Optional. Array-like of shape (n, 2) specifying lower and
        upper bounds for each variable.
    :return: The wrapped function.
    """
    def wrapper(*args, **kwargs):
        wrapper.calls += 1
        result = func(*args, **kwargs)

        # Check bounds if provided and first arg is an array-like
        if bounds is not None and len(args) > 0:
            x = np.asarray(args[0])
            b = np.asarray(bounds)

            # If invalid input, log infinity
            if not (np.all(x >= b[:, 0]) and np.all(x <= b[:, 1])):
                wrapper.log.append((wrapper.calls, np.inf))
            else:
                wrapper.log.append((wrapper.calls, result))
        else:
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

        * ``min``: The minimum value function value, or None if unknown.
        * ``log``: Solver log in form [(fcalls, fvals)].
        * ``problem``: The problem class being tested.
        * ``n_dims``: Number of dimensions for the problem.

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

def generate_initial_conditions(bounds, n_points, constraints=None, seed=None):
    """
    Generate initial conditions as needed by the optimizer.

    :param bounds: Problem bounds as numpy array
    :param n_points: Number of initial points to generate
    :param constraints: List of constraints to satisfy.
    :param seed: Random seed for reproducibility
    :return: Array of initial points
    """
    if seed is not None:
        np.random.seed(seed)

    if n_points == 0:
        # No initial conditions needed
        return None
    if not constraints:
        if n_points == 1:
            # Single initial condition
            return np.random.uniform(bounds[:, 0], bounds[:, 1], size=(bounds.shape[0],))
        else:
            # Multiple initial conditions
            return np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_points, bounds.shape[0]))
    else:
        # Generate initial conditions that satisfy constraints
        initial_conditions = []
        attempts = 0
        while len(initial_conditions) < n_points and attempts < 1000000:
            x = np.random.uniform(bounds[:, 0], bounds[:, 1])
            if all(constraint(x) >= 0 for constraint in constraints):
                initial_conditions.append(x)
            attempts += 1

        if len(initial_conditions) < n_points:
            warnings.warn(f"Only {len(initial_conditions)} valid initial conditions found after {attempts} attempts.")

        return np.array(initial_conditions)

def run_standard(solvers, problems, test_dimensions=[2, 4, 5, 8, 10, 12], n_iters=5, output_folder=None, track_energy=True, verbose=False):
    """
    Run a list of solvers on a set of problems from pyGOLD's standard problems
    module and generate log files in the COCO format. To postprocess the results
    with COCOPP, the problems must be n-dimensional and test_dimensions must
    have exactly 6 dimensions (see the configure_testbed function docs). For
    other configurations, use the postprocessing functions presented in
    pygold.postprocessing.

    Each tested solver must be a subclass of BaseOptimizer, providing the
    attributes

    * ``deterministic`` (bool):
        If the algorithm is deterministic (True) or stochastic (False)

    * ``n_points`` (int):
        The number of initial points the algorithm requires.

    * ``optimize`` (callable):
        A function that takes arguments ``func`` (the
        objective function), ``bounds`` (as a list of tuples),
        ``x0`` (the initial point(s)), ``constraints`` (constraint functions,
        which return a negative value if violated), and ``**kwargs`` and returns
        an `OptimizationResult` object. The ``optimize`` method should handle
        the optimization process, including any necessary preprocessing of the
        problem or input data as required by the specific optimization
        algorithm. The optimize method will be passed an empty list for
        ``constraints`` if the test problem is unconstrained.

    See the tutorials on GitHub for examples.

    Performance data is recorded to output_data/ in the COCO format, which
    includes the number of function evaluations and the difference between the
    solution and the best known minimum. If the true function minimum is
    unknown, the smallest calculated function value is used as the best known
    minimum.

    If the `track_energy` parameter is set to True, the energy consumption
    of each solver is also tracked and saved in
    output_data/energy_data_standard.csv.

    :param solvers: List of solver instances implemented as subclasses of
        BaseOptimizer.
    :param problems: List of problem classes from the standard problems module.
    :param test_dimensions: List of dimensions to test any n-dimensional
        problems on, defaults to ``[2, 4, 6, 8, 10, 12]``.
    :param n_iters: Number of runs for each problem. Each solver will be run
        ``n_iters`` times on each problem, with different random seeds
        per run consistent across solvers, defaults to ``5``.
    :param output_folder: Folder to save the output data. Defaults to
        ``output_data``.
    :param track_energy: If True, track the energy consumption of each solver,
        defaults to ``True``.
    :param verbose: If True, prints progress of the run, defaults to ``False``.
    """
    if output_folder is None:
        output_folder = "output_data"

    if track_energy:
        if not CODECARBON_AVAILABLE:
            warnings.warn("CodeCarbon is not available, energy tracking will not be performed.")
            track_energy = False
        else:
            os.makedirs(output_folder, exist_ok=True)
            # Initialize tracker
            tracker = EmissionsTracker(
                project_name="pygold_standard_problems",
                output_dir=output_folder,
                save_to_file=False,
                measure_power_secs=0.5,
                log_level="error"
            )

            energy_results = pd.DataFrame()

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

            orig_eval = prob.evaluate

            for solver in solvers:
                # Instantiate the solver if it's a class
                if isinstance(solver, type):
                    solver = solver()

                # Check that solver is a subclassed from BaseOptimizer or
                # supplies deterministic and n_points attributes and a
                # function that takes arguments func, bounds, x0, and **kwargs
                # and returns a OptimizationResult object
                if isinstance(solver, BaseOptimizer):
                    pass
                elif callable(solver.optimize):
                    sig = signature(solver.optimize)
                    if 'func' not in sig.parameters or 'bounds' not in sig.parameters or 'x0' not in sig.parameters:
                        raise ValueError(f"Solver {solver} does not have the required signature for optimization.")
                    if 'n_points' not in solver.__dict__:
                        raise ValueError(f"Solver {solver} does not define an 'n_points' attribute.")
                else:
                    raise ValueError(f"Solver {solver} is not a valid optimizer derived from the BaseOptimizer class.")

                # Take data from the solver
                n_points = solver.n_points
                deterministic = solver.deterministic

                # At this point, solver is guaranteed to be callable and
                # initialization_type and deterministic are defined
                solver_name = solver.__class__.__name__

                # Figure out how many iterations to run
                if deterministic and n_points == 0:
                    # Run once without initial conditions
                    iterations = 1
                else:
                    # Run multiple times with different initial points
                    iterations = n_iters

                for i in range(iterations):
                    np.random.seed(i)  # Ensure reproducibility between solvers

                    # Wrap the problem with a logger
                    prob.evaluate = logger(orig_eval, prob.bounds())

                    # Generate initial conditions based on algorithm type
                    bounds = np.array(prob.bounds())
                    bounds[np.isneginf(bounds)] = -2000
                    bounds[np.isposinf(bounds)] = 2000

                    initial_conditions = generate_initial_conditions(bounds, n_points, prob.constraints(), seed=i)

                    if verbose:
                        print(f"Running {solver_name} on {problem.__name__} in {n_dims}D, iteration {i+1}/{iterations}")

                    if track_energy:
                        task_name = f"{solver_name}_{problem.__name__}_{n_dims}D_iter{i}"
                        tracker.start_task(task_name)

                    try:
                        # Run solver on problem
                        # Passes to solvers optimize function the objective
                        # function, problem bounds, initial conditions (may be
                        # None), and list of constraint functions (may be empty)
                        res = solver.optimize(prob.evaluate, bounds, initial_conditions, prob.constraints())

                    except Exception as e:
                        warnings.warn(f"Solver {solver_name} failed on {problem.__name__} with dimensions {n_dims}, run {i+1}/{iterations}: {e}")
                        if track_energy:
                            tracker.stop_task(task_name)
                        continue

                    finally:
                        if track_energy:
                            tmp = tracker.stop_task(task_name)
                            row_data = tmp.values
                            row_data['solver'] = solver_name
                            row_data['problem'] = problem.__name__
                            row_data['n_dims'] = n_dims
                            row_data['instance'] = i
                            energy_results = pd.concat([energy_results, pd.DataFrame([row_data])], ignore_index=True)

                    if prob.constraints() and res.x is not None:
                        # Check if the solution satisfies the constraints
                        if any(constraint(res.x) < -1e-6 for constraint in prob.constraints()):
                            warnings.warn(f"Solver {solver_name} returned a solution ({res.x}) that does not satisfy constraints for {problem.__name__} in {n_dims}D, run {i+1}/{iterations}.")
                            continue

                    results.append({'solver': solver_name,
                                    'problem': problem.__name__,
                                    'func_id': id,
                                    'instance': i,
                                    'n_dims': n_dims,
                                    'min': prob.min(),
                                    'log': prob.evaluate.log,
                                    })

            # Results for this problem and dimension are now complete
            # Resolve unknown min case
            results = resolve_unknown_min(results)

            # Save results to file in COCO format
            log_coco_from_results(results, output_folder=output_folder)

    # Stop the tracker at the end of all standard problem runs
    if track_energy:
        tracker.stop()
        # save data to CSV
        folder_path = os.path.join(output_folder, "energy_data_standard.csv")
        energy_results.to_csv(folder_path, index=False)

def run_floris(solvers, problems, n_turbines=[2, 4, 5, 8, 10, 12], n_iters=5, output_folder=None, track_energy=True, verbose=False):
    """
    Run a list of solvers on a set of problems from the FLORIS module
    and generate log files in the COCO format. The FLORIS problems are
    maximization problems, so the objective function is inverted to
    make them minimization problems. The problems are expected to be
    supplied un-altered.

    To postprocess the results with COCOPP, the problems must be n-dimensional
    and test_dimensions must have exactly 6 dimensions (see the
    configure_testbed function docs). For other configurations, use the
    postprocessing functions presented in pygold.postprocessing.

    Each tested solver must be a subclass of BaseOptimizer, providing the
    attributes

    * ``deterministic`` (bool):
        If the algorithm is deterministic (True) or stochastic (False)

    * ``n_points`` (int):
        The number of initial points the algorithm requires.

    * ``optimize`` (callable):
        A function that takes arguments ``func`` (the
        objective function), ``bounds`` (as a list of tuples),
        ``x0`` (the initial point(s)), ``constraints`` (constraint functions,
        which return a negative value if violated), and ``**kwargs`` and returns
        an `OptimizationResult` object. The ``optimize`` method should handle
        the optimization process, including any necessary preprocessing of the
        problem or input data as required by the specific optimization
        algorithm. The optimize method will be passed an empty list for
        ``constraints`` if the test problem is unconstrained.

    See the tutorials on GitHub for examples.

    Performance data is recorded to output_data/ in the COCO format, which
    includes the number of function evaluations and the difference between the
    solution and the best known minimum. If the true function minimum is
    unknown, the smallest calculated function value is used as the best known
    minimum.

    If the `track_energy` parameter is set to True, the energy consumption
    of each solver is also tracked and saved in
    output_data/energy_data_floris.csv.

    :param solvers: List of solver instances implemented as subclasses of
        BaseOptimizer.
    :param problems: List of problem classes from the FLORIS problems module.
    :param n_turbines: List of turbine counts to test, defaults to
        ``[2, 4, 5, 8, 10, 12]``.
    :param n_iters: Number of runs for each problem. Each solver will be run
        ``n_iters`` times on each problem, with different random seeds
        per run consistent across solvers, defaults to ``5``.
    :param output_folder: Folder to save the output data. Defaults to
        ``output_data``.
    :param track_energy: If True, track the energy consumption of each solver,
        defaults to ``True``.
    :param verbose: If True, prints progress of the run, defaults to ``False``.
    """
    if output_folder is None:
        output_folder = "output_data"

    if track_energy:
        if not CODECARBON_AVAILABLE:
            warnings.warn("CodeCarbon is not available, energy tracking will not be performed.")
            track_energy = False
        else:
            os.makedirs(output_folder, exist_ok=True)
            # Initialize tracker
            tracker = EmissionsTracker(
                project_name="pygold_floris_problems",
                output_dir=output_folder,
                save_to_file=False,
                measure_power_secs=0.5,
                log_level="error"
            )

            energy_results = pd.DataFrame()

    for id, problem in enumerate(problems):

        for n_turb in n_turbines:
            results = []
            prob = problem(n_turb)

            orig_eval = prob.evaluate

            for solver in solvers:
                # Instantiate the solver if it's a class
                if isinstance(solver, type):
                    solver = solver()

                # Check that solver is a subclassed from BaseOptimizer or
                # supplies deterministic and n_points attributes and a
                # function that takes arguments func, bounds, x0, and **kwargs
                # and returns a OptimizationResult object
                if isinstance(solver, BaseOptimizer):
                    pass
                elif callable(solver.optimize):
                    sig = signature(solver.optimize)
                    if 'func' not in sig.parameters or 'bounds' not in sig.parameters or 'x0' not in sig.parameters:
                        raise ValueError(f"Solver {solver} does not have the required signature for optimization.")
                    if 'n_points' not in solver.__dict__:
                        raise ValueError(f"Solver {solver} does not define an 'n_points' attribute.")
                else:
                    raise ValueError(f"Solver {solver} is not a valid optimizer derived from the BaseOptimizer class.")

                # Take data from the solver
                n_points = solver.n_points
                deterministic = solver.deterministic

                # At this point, solver is guaranteed to be callable and
                # initialization_type and deterministic are defined
                solver_name = solver.__class__.__name__

                # Figure out how many iterations to run
                if deterministic and n_points == 0:
                    # Run once without initial conditions
                    iterations = 1
                else:
                    # Run multiple times with different initial points
                    iterations = n_iters

                for i in range(iterations):
                    np.random.seed(i)  # Ensure reproducibility between solvers

                    # Invert the objective to make it a minimization problem
                    # And wrap the problem with a logger
                    prob.evaluate = logger(lambda *args, **kwargs: -orig_eval(*args, **kwargs), bounds=prob.bounds())

                    # Generate initial layout
                    attempts = 0
                    while attempts < 1000000:
                        # Generate random layout
                        x = np.random.uniform(0, 1000, size=(n_turb, 2))

                        # Sort turbines to satisfy permutation constraint
                        x = x[np.argsort(x[:, 0])]

                        # Check constraints
                        dist_constraints = prob.dist_constraint(x.flatten())
                        perm_constraints = prob.perm_constraint(x.flatten())
                        if np.all(dist_constraints >= 0) and np.all(perm_constraints >= 0):
                            break
                        attempts += 1

                    if attempts == 1000000:
                        warnings.warn(f"Failed to generate valid initial layout for {problem.__name__} with {n_turb} turbines after 1000000 attempts. Skipping solver run.")
                        continue

                    if prob.DIM == 3:
                        # Add random yaw angles close to zero
                        x = np.hstack((x, np.random.uniform(-np.pi/32, np.pi/32, size=(n_turb, 1))))

                    x = x.flatten()

                    if verbose:
                        print(f"Running {solver_name} on {problem.__name__} with {n_turb} turbines, {n_turb * problem.DIM} dimensions, iteration {i+1}/{iterations}")

                    if track_energy:
                        task_name = f"{solver_name}_{problem.__name__}_{n_turb}turb_iter{i}"
                        tracker.start_task(task_name)

                    try:
                        res = solver.optimize(prob.evaluate, prob.bounds(), x, prob.constraints())

                    except Exception as e:
                        warnings.warn(f"Solver {solver_name} failed on {problem} with {n_turb} turbines, {n_turb * problem.DIM} dimensions, run {i+1}/{iterations}: {e}")
                        if track_energy:
                            tracker.stop_task(task_name)
                        continue

                    finally:
                        if track_energy:
                            tmp = tracker.stop_task(task_name)
                            row_data = tmp.values
                            row_data['solver'] = solver_name
                            row_data['problem'] = problem.__name__
                            row_data['n_dims'] = n_turb * problem.DIM
                            row_data['instance'] = i
                            energy_results = pd.concat([energy_results, pd.DataFrame([row_data])], ignore_index=True)

                    if prob.constraints() and res.x is not None:
                        # Check if the solution satisfies the constraints
                        if np.any([np.any(constraint(res.x) < -1e-6) for constraint in prob.constraints()]):
                            warnings.warn(f"Solver {solver_name} returned a solution that does not satisfy constraints for {problem.__name__}, run {i+1}/{iterations}.")
                            continue

                    results.append({'solver': solver_name,
                                    'problem': problem.__name__,
                                    'func_id': id,
                                    'instance': i,
                                    'n_dims': n_turb * problem.DIM,
                                    'min': None,
                                    'log': prob.evaluate.log,
                                    })
            # Results for this problem and dimension are now complete
            # Resolve unknown min case
            results = resolve_unknown_min(results)

            # Save results to file in COCO format
            log_coco_from_results(results, output_folder=output_folder)

    # Stop the tracker at the end of all FLORIS problem runs
    if track_energy:
        tracker.stop()
        # save data to CSV
        folder_path = os.path.join(output_folder, "energy_data_floris.csv")
        energy_results.to_csv(folder_path, index=False)

def run_solvers(solvers, problems, test_dimensions=[2, 4, 5, 8, 10, 12], n_iters=5, output_folder=None, track_energy=True, verbose=False):
    """
    Run a list of solvers on a set of problems and generate log files in the
    COCO format. To postprocess the results with COCOPP, the problems must be
    n-dimensional and test_dimensions must have exactly 6 dimensions (see the
    configure_testbed function docs). For other configurations, use the
    postprocessing functions presented in pygold.postprocessing.

    Each tested solver must be a subclass of BaseOptimizer, providing the
    attributes

    * ``deterministic`` (bool):
        If the algorithm is deterministic (True) or stochastic (False)

    * ``n_points`` (int):
        The number of initial points the algorithm requires.

    * ``optimize`` (callable):
        A function that takes arguments ``func`` (the
        objective function), ``bounds`` (as a list of tuples),
        ``x0`` (the initial point(s)), ``constraints`` (constraint functions,
        which return a negative value if violated), and ``**kwargs`` and returns
        an `OptimizationResult` object. The ``optimize`` method should handle
        the optimization process, including any necessary preprocessing of the
        problem or input data as required by the specific optimization
        algorithm. The optimize method will be passed an empty list for
        ``constraints`` if the test problem is unconstrained.

    See the tutorials on GitHub for examples.

    Problems can be from the standard problems module or the Floris module.
    Floris problems use test_dimensions to specify the number of turbines
    to test. If a problem is not from either module, the standard problem
    runner is used.

    Data is recorded to output_data/ in the COCO format, which includes the
    number of function evaluations and the difference between the solution and
    the best known minimum. If the true function minimum is unknown, the
    smallest calculated function value is used as the best known minimum.

    If the `track_energy` parameter is set to True, the energy consumption
    of each solver is also tracked and saved in output_data/energy_data.csv.

    :param solvers: List of solver instances.
    :param problems: List of problem classes.
    :param test_dimensions: List of dimensions to test any n-dimensional
        problems on, defaults to ``[2, 4, 6, 8, 10, 12]``.
    :param n_iters: Number of runs for each problem. Each solver will be run
        ``n_iters`` times on each problem, with different random seeds
        per run consistent across solvers, defaults to ``5``.
    :param output_folder: Folder to save the output data. Defaults to
        ``output_data``.
    :param track_energy: If True, track the energy consumption of each solver,
        defaults to ``True``.
    :param verbose: If True, prints progress of the run, defaults to ``False``.
    """
    if output_folder is None:
        output_folder = "output_data"

    standard_problems = []
    floris_problems = []
    for p in problems:
        if p.__module__.startswith('pygold.problems.standard'):
            standard_problems.append(p)
        elif p.__module__.startswith('pygold.problems.floris'):
            floris_problems.append(p)
        else:
            warnings.warn(f"Problem {p.__name__} is not a standard or FLORIS problem, trying to use the standard problem runner.")
            standard_problems.append(p)

    if len(standard_problems) > 0:
        run_standard(solvers, standard_problems, test_dimensions=test_dimensions, n_iters=n_iters, output_folder=output_folder, track_energy=track_energy, verbose=verbose)

    if len(floris_problems) > 0:
        run_floris(solvers, floris_problems, n_turbines=test_dimensions, n_iters=n_iters, output_folder=output_folder, track_energy=track_energy, verbose=verbose)

    if track_energy:
        if not CODECARBON_AVAILABLE:
            warnings.warn("CodeCarbon is not available, energy tracking will not be performed.")
            return
        # Combine the energy results into one file
        std_file = os.path.join(output_folder, "energy_data_standard.csv")
        floris_file = os.path.join(output_folder, "energy_data_floris.csv")
        combined_file = os.path.join(output_folder, "energy_data.csv")
        if os.path.exists(std_file) and os.path.exists(floris_file):
            energy_results = pd.concat([pd.read_csv(std_file), pd.read_csv(floris_file)], ignore_index=True)
            energy_results.to_csv(combined_file, index=False)
        elif os.path.exists(std_file):
            os.rename(std_file, combined_file)
        elif os.path.exists(floris_file):
            os.rename(floris_file, combined_file)
