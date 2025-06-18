from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np


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
        - ``success`` (*bool*): If the solver successfully solved the problem.
        - ``metric`` (*float*): Value of the evaluation metric (fxn evals).

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
