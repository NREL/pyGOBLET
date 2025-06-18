import scipy.optimize as opt
import numpy as np
import pygold
import pygold.problems.standard_problems as bp

def workflow(solvers, problems, test_dimensions, n_times, verbose=True):
    """
    Run the workflow to benchmark SciPy optimizers on benchmark functions.
    """
    # Run solvers
    results_fxn_evals = pygold.default_runners.run_solvers_fxn_evals(solvers, problems=problems, test_dimensions=test_dimensions, n_runs=n_times, verbose=verbose)

    # Deal with problems with no known minimum
    # Best found problem value is treated as the minimum
    results_fxn_evals = pygold.default_runners.resolve_unknown_min_old(results_fxn_evals)

    # Compute performance ratios
    ratios_fxn_evals = pygold.compute_performance_ratios(results_fxn_evals)

    # Compute performance profiles
    profiles_fxn_evals = pygold.compute_performance_profiles(ratios_fxn_evals, tau_grid=np.linspace(1, 2250, 100))

    # Plot performance profiles
    pygold.plot_performance_profiles([profiles_fxn_evals], metrics=['Function evaluations'])

# Choose solvers to benchmark - Unconstrained global optimizers
solvers = [opt.basinhopping, opt.differential_evolution, opt.dual_annealing, opt.shgo]

# Choose test functions
problems = bp.__Unconstrained__
test_dimensions = [2, 5]
n_times = 2

# Run solvers
workflow(solvers, problems, test_dimensions, n_times)

## SPLIT CONTRAINTED AND UNCONSTRAINED INTO TWO FILES ##
problems = bp.__Constrained__
n_times = 10
# Choose solvers to benchmark - Constrained global optimizers
constrained_solvers = [opt.shgo, opt.basinhopping]

# Choose constrained test problems
problems = bp.__Constrained__

# Run solvers
workflow(constrained_solvers, problems, test_dimensions, n_times, verbose=True)
