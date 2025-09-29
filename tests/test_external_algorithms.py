import numpy as np
import pytest
import pygoblet

# Placeholder algorithm implementations
def algorithm_a(func, bounds, maxevals, seed=None):
    np.random.seed(seed)
    # ... actual optimization logic ...
    return np.zeros(len(bounds)), func(np.zeros(len(bounds)))

def algorithm_b(func, bounds, maxevals, seed=None):
    np.random.seed(seed)
    # ... actual optimization logic ...
    return np.ones(len(bounds)), func(np.ones(len(bounds)))

def algorithm_c(func, bounds, maxevals, seed=None):
    np.random.seed(seed)
    # ... actual optimization logic ...
    return np.full(len(bounds), 2.0), func(np.full(len(bounds), 2.0))

# Select three representative problems from pyGOBLET
problems = [
    pygoblet.get_standard_problems("Bowl_shaped")[0],      # e.g., Sphere
    pygoblet.get_standard_problems("Plate_shaped")[0],     # e.g., Schwefel
    pygoblet.get_standard_problems("Many_local_minima")[0] # e.g., Rastrigin
]

algorithms = [algorithm_a, algorithm_b, algorithm_c]
algorithm_names = ["algorithm_a", "algorithm_b", "algorithm_c"]

@pytest.mark.parametrize("alg,alg_name", zip(algorithms, algorithm_names))
def test_algorithms_on_problems(alg, alg_name):
    n_runs = 5
    maxevals = 2000
    tol = 1e-2  # Acceptable tolerance from known minimum
    min_success_rate = 0.6  # Require at least 60% of runs to succeed
    results = {}
    for problem in problems:
        # Initialize problem in 5D for consistency
        prob_instance = problem(5)
        func = lambda x: problem.evaluate(x)
        bounds = prob_instance.bounds()
        min_value = problem.min()
        run_vals = []
        for run in range(n_runs):
            x_best, f_best = alg(func, bounds, maxevals, seed=run)
            run_vals.append(f_best)
        run_vals = np.array(run_vals)
        n_success = np.sum(np.abs(run_vals - min_value) < tol)
        success_rate = n_success / n_runs
        results[problem.__name__] = success_rate
        assert success_rate >= min_success_rate, (
            f"{alg_name} failed on {problem.__name__}: success rate {success_rate:.2f} < {min_success_rate}"
        )
    print(f"{alg_name} success rates:", results)
