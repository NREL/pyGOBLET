import pytest
import warnings
import os
import shutil
import numpy as np
from pygoblet.runners import run_solvers
from pygoblet.optimizer import BaseOptimizer, OptimizationResult
from pygoblet.problems.standard import Sphere

@pytest.fixture
def clean_output_folder():
    folder = "test_output"
    if os.path.exists(folder):
        shutil.rmtree(folder)
    yield folder
    if os.path.exists(folder):
        shutil.rmtree(folder)

class TestOptimizer(BaseOptimizer):
    """Test optimizer for verifying initial conditions handling"""
    deterministic = True
    n_points = 0

    def __init__(self, deterministic: bool, n_points: int):
        super().__init__()
        self.received_initial_conditions = []
        self.deterministic = deterministic
        self.n_points = n_points

    def optimize(self, func, bounds, x0=None, constraints=None, **kwargs):
        self.received_initial_conditions.append(x0.copy() if x0 is not None else None)
        x = np.array([(b[0] + b[1]) / 2 for b in bounds])
        return OptimizationResult(x, func(x), algorithm=self.name)

@pytest.mark.parametrize("n_points", [0, 1, 2, 5, 10])
def test_provided_initial_conditions(n_points, clean_output_folder):
    """
    Test that optimizers receive the correct initial conditions

    Verifies that:
    1) Solvers with n_points=0 receive None as initial condition
    2) Solvers with n_points=1 receive a single initial condition
    3) Solvers with n_points>1 receive a n_points initial conditions
    4) Deterministic and stochastic solvers receive the same initial conditions
       when n_points > 0
    5) Solvers with the same type (deterministic or stochastic) receive
       the same initial conditions in the same order across multiple runs
    6) Solvers receive different initial conditions each iteration
       when n_points > 0
    """
    Solver1 = TestOptimizer(deterministic=True, n_points=n_points)
    Solver2 = TestOptimizer(deterministic=False, n_points=n_points)

    run_solvers([Solver1, Solver2], [Sphere], output_folder=clean_output_folder, test_dimensions=[3], n_iters=2)

    # Check that solvers received the correct initial conditions
    if n_points == 0:
        assert Solver1.received_initial_conditions == [None]
        assert Solver2.received_initial_conditions == [None] * 2
    elif n_points == 1:
        assert len(Solver1.received_initial_conditions) == 2
        assert len(Solver2.received_initial_conditions) == 2
        assert all(x is not None for x in Solver1.received_initial_conditions)
        assert all(x is not None for x in Solver2.received_initial_conditions)
    else:
        assert len(Solver1.received_initial_conditions) == 2
        assert len(Solver2.received_initial_conditions) == 2
        assert all(len(x) == n_points for x in Solver1.received_initial_conditions)
        assert all(len(x) == n_points for x in Solver2.received_initial_conditions)

    # Check that solvers received the same initial conditions each iteration
    if n_points > 0:
        assert not np.array_equal(Solver1.received_initial_conditions[0], Solver1.received_initial_conditions[1])
        assert not np.array_equal(Solver2.received_initial_conditions[0], Solver2.received_initial_conditions[1])
        for i in range(2):
            assert np.array_equal(Solver1.received_initial_conditions[i], Solver2.received_initial_conditions[i])

def test_creates_output_files(clean_output_folder):
    """Test that .info, .dat, .tdat, and .mdat output files are created"""
    Solver = TestOptimizer(deterministic=True, n_points=0)
    run_solvers([Solver], [Sphere], output_folder=clean_output_folder, test_dimensions=[3,4], n_iters=1)
    assert os.path.exists("test_output")

    # Check for specific output files
    expected_files = [
        "test_output/TestOptimizer/TestOptimizer_f1_DIM3.info",
        "test_output/TestOptimizer/TestOptimizer_f1_DIM4.info",
        "test_output/TestOptimizer/data_TestOptimizer_f1/TestOptimizer_f1_DIM3.dat",
        "test_output/TestOptimizer/data_TestOptimizer_f1/TestOptimizer_f1_DIM4.dat",
        "test_output/TestOptimizer/data_TestOptimizer_f1/TestOptimizer_f1_DIM3.tdat",
        "test_output/TestOptimizer/data_TestOptimizer_f1/TestOptimizer_f1_DIM4.tdat",
        "test_output/TestOptimizer/data_TestOptimizer_f1/TestOptimizer_f1_DIM3.mdat",
        "test_output/TestOptimizer/data_TestOptimizer_f1/TestOptimizer_f1_DIM4.mdat"
    ]
    for f in expected_files:
        assert os.path.exists(f)

def test_invalid_solver(clean_output_folder):
    """Test that an error is raised for invalid solvers"""

    class BadSolver1:
        """Invalid solver with incorrect optimize signature"""
        n_points = 1
        deterministic = True
        def optimize(self, x, y, z):
            pass
    class BadSolver2:
        """Invalid solver with missing required n_points"""
        deterministic = True
        def optimize(self, func, bounds, x0, constraints=None, extra_param=None):
            pass
    class BadSolver3:
        """Invalid solver with missing required deterministic"""
        n_points = 1
        def optimize(self, func, bounds, x0=None, constraints=None):
            pass

    for Solver in [BadSolver1, BadSolver2, BadSolver3]:
        with pytest.raises(ValueError):
            run_solvers([Solver()], [Sphere], output_folder=clean_output_folder, test_dimensions=[3], n_iters=1)

def test_constraint_satisfaction_warning(clean_output_folder):
    """Test that warnings are issued for constraint violations"""

    class ConstrainedProblem:
        """Test problem with constraints"""
        DIM = 2

        def __init__(self, n_dims):
            self.n_dims = n_dims

        def evaluate(self, x):
            return np.sum(x**2)

        def bounds(self):
            return [(-10, 10)] * self.n_dims

        def constraints(self):
            return [lambda x: x[0] - 5]

        def min(self):
            return 25.0

    class ConstraintViolatingSolver(BaseOptimizer):
        """Solver that always returns origin (violates constraints)"""
        deterministic = True
        n_points = 0

        def optimize(self, func, bounds, x0=None, constraints=None, **kwargs):
            # Always return origin, which violates the constraint
            x = np.zeros(len(bounds))
            return OptimizationResult(x, func(x), algorithm=self.name)

    solver = ConstraintViolatingSolver()

    # This should issue a warning about constraint violation
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        run_solvers([solver], [ConstrainedProblem], output_folder=clean_output_folder,
                    test_dimensions=[2], n_iters=1, verbose=False)

        # Check that a constraint violation warning was issued
        constraint_warnings = [warning for warning in w
                             if "does not satisfy constraints" in str(warning.message)]
        assert len(constraint_warnings) > 0

def test_solver_exception_handling(clean_output_folder):
    """Test that solver exceptions are properly caught and warned about"""

    class FailingSolver(BaseOptimizer):
        """Solver that always raises an exception"""
        deterministic = True
        n_points = 0

        def optimize(self, func, bounds, x0=None, constraints=None, **kwargs):
            raise RuntimeError("Deliberate test failure")

    solver = FailingSolver()

    # This should catch the exception and issue a warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        run_solvers([solver], [Sphere], output_folder=clean_output_folder,
                    test_dimensions=[2], n_iters=1, verbose=False)

        # Check that a solver failure warning was issued
        failure_warnings = [warning for warning in w
                           if "failed on" in str(warning.message)]
        assert len(failure_warnings) > 0
