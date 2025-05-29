import numpy as np
from global_opt_bench import benchmark_tools as bt

def test_compute_performance_ratios():
    # Mock results: 2 solvers, 3 problems, 2 iterations/random seeds each
    results = [
        # Problem P1
        {'solver': 'A', 'problem': 'P1', 'random_seed': 0, 'n_dims': 2, 'success': True,  'metric': 2.0},
        {'solver': 'B', 'problem': 'P1', 'random_seed': 0, 'n_dims': 2, 'success': True,  'metric': 4.0},
        {'solver': 'A', 'problem': 'P1', 'random_seed': 1, 'n_dims': 2, 'success': True,  'metric': 2.5},
        {'solver': 'B', 'problem': 'P1', 'random_seed': 1, 'n_dims': 2, 'success': True,  'metric': 3.0},
        # Problem P2
        {'solver': 'A', 'problem': 'P2', 'random_seed': 0, 'n_dims': 2, 'success': False, 'metric': 3.0},
        {'solver': 'B', 'problem': 'P2', 'random_seed': 0, 'n_dims': 2, 'success': True,  'metric': 6.0},
        {'solver': 'A', 'problem': 'P2', 'random_seed': 1, 'n_dims': 2, 'success': True,  'metric': 5.0},
        {'solver': 'B', 'problem': 'P2', 'random_seed': 1, 'n_dims': 2, 'success': False, 'metric': 7.0},
        # Problem P3
        {'solver': 'A', 'problem': 'P3', 'random_seed': 0, 'n_dims': 2, 'success': False,  'metric': 1.5},
        {'solver': 'B', 'problem': 'P3', 'random_seed': 0, 'n_dims': 2, 'success': False, 'metric': 2.0},
        {'solver': 'A', 'problem': 'P3', 'random_seed': 1, 'n_dims': 2, 'success': True, 'metric': 5.4},
        {'solver': 'B', 'problem': 'P3', 'random_seed': 1, 'n_dims': 2, 'success': True,  'metric': 1.8},
    ]

    expected = {
        ('P1', 2, 0): {'A': 1.0, 'B': 2.0},
        ('P1', 2, 1): {'A': 1.0, 'B': 1.2},
        ('P2', 2, 0): {'A': float('inf'), 'B': 1.0},
        ('P2', 2, 1): {'A': 1.0, 'B': float('inf')},
        ('P3', 2, 0): {'A': float('inf'), 'B': float('inf')},
        ('P3', 2, 1): {'A': 3.0, 'B': 1.0},
    }

    ratios = bt.compute_performance_ratios(results)

    # Compare ratios and expected, handling float('inf') and floats
    assert ratios.keys() == expected.keys()
    for k in expected:
        assert ratios[k].keys() == expected[k].keys()
        for solver in expected[k]:
            v1 = ratios[k][solver]
            v2 = expected[k][solver]
            if np.isinf(v1) and np.isinf(v2):
                continue
            else:
                assert np.isclose(v1, v2, rtol=1e-8), f"Mismatch at {k}, {solver}: {v1} != {v2}"

def test_compute_performance_profiles():
    # Mock ratios for 2 problems, 2 solvers
    ratios = {
        ('P1', 2, 0): {'A': 1.0, 'B': 2.0},
        ('P1', 2, 1): {'A': 1.0, 'B': 1.2},
        ('P2', 2, 0): {'A': float('inf'), 'B': 1.0},
        ('P2', 2, 1): {'A': 1.0, 'B': float('inf')},
        ('P3', 2, 0): {'A': float('inf'), 'B': float('inf')},
        ('P3', 2, 1): {'A': 3.0, 'B': 1.0},
    }

    tau_grid = np.array([1, 2, 10])
    profiles = bt.compute_performance_profiles(ratios, tau_grid)

    np.testing.assert_allclose(profiles['A'][1], [0.5, 0.5, 2/3])
    np.testing.assert_allclose(profiles['B'][1], [1/3, 2/3, 2/3])
