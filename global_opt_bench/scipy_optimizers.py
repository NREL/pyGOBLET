import scipy.optimize as opt
import numpy as np
import benchmark_tools as bt
import benchmark_functions as bf

# Choose solvers to benchmark
solvers = [opt.basinhopping, opt.differential_evolution, opt.dual_annealing, opt.shgo]

# Choose test functions
problems = bf.__All__
test_dimensions = [2, 5]
n_times = 2

# Run solvers
results_time = bt.run_solvers_time(solvers, problems=problems, test_dimensions=test_dimensions, n_runs=n_times, verbose=True)
results_fxn_evals = bt.run_solvers_fxn_evals(solvers, problems=problems, test_dimensions=test_dimensions, n_runs=n_times, verbose=True)

# Compute performance ratios
ratios_time = bt.compute_performance_ratios(results_time)
ratios_fxn_evals = bt.compute_performance_ratios(results_fxn_evals)

# Compute performance profiles
profiles_time = bt.compute_performance_profiles(ratios_time, tau_grid=np.linspace(1, 2250, 100))
profiles_fxn_evals = bt.compute_performance_profiles(ratios_fxn_evals, tau_grid=np.linspace(1, 2250, 100))

# Plot performance profiles
bt.plot_performance_profiles([profiles_time, profiles_fxn_evals], metrics=['Solve time', 'Function evaluations'])

print()
