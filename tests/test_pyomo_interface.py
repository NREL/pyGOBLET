import pytest
import inspect
import pygold.benchmark_functions as bf
import numpy as np
try:
    import pyomo.environ as pyo
except ImportError:
    pyo = None

# List of benchmark function classes to test (add more as needed)
FUNCTION_CLASSES = [
    obj for name, obj in inspect.getmembers(bf, inspect.isclass)
    if obj.__module__ == bf.__name__ and obj not in (bf.BenchmarkFunction, bf.className)
]

def get_test_points(bounds):
    points = []
    for b in bounds:
        if b[0] == float("-inf") and b[1] == float("inf"):
            b[0] = -1.0
            b[1] = 1.0
        elif b[0] == float("-inf"):
            b[0] = -1.0
        elif b[1] == float("inf"):
            b[1] = 1.0
    # Lower bound
    points.append([b[0] for b in bounds])
    # Upper bound
    points.append([b[1] for b in bounds])
    # Midpoint
    points.append([0.5 * (b[0] + b[1]) for b in bounds])
    # Random point (seeded for reproducibility)
    rng = np.random.default_rng(42)
    points.append([float(rng.uniform(b[0], b[1])) for b in bounds])
    return points

# Test that all benchmark function classes have pyomo model conversion
@pytest.mark.parametrize("cls", FUNCTION_CLASSES)
def test_as_pyomo_model(cls):
    if pyo is None:
        pytest.skip("Pyomo not installed")
    # Determine a valid dimension for the class
    dim = getattr(cls, "DIM", 2)
    if isinstance(dim, int):
        n = dim
    elif isinstance(dim, tuple):
        n = dim[0]
    # Instantiate and test
    instance = cls(n)
    model = instance.as_pyomo_model()
    assert hasattr(model, "x")
    assert hasattr(model, "obj")
    assert isinstance(model, pyo.ConcreteModel)
    # Test that the objective can be evaluated numerically at initial values
    try:
        val = pyo.value(model.obj)
        assert isinstance(val, float)
    except Exception as e:
        pytest.fail(f"Objective not numerically evaluable: {e}")

# Test that all pyomo models match the numpy model
@pytest.mark.parametrize("func_cls", FUNCTION_CLASSES)
def test_pyomo_expr_matches_numpy(func_cls):
    if pyo is None:
        pytest.skip("Pyomo not installed")
    # Instantiate with default dimension
    try:
        func = func_cls()
    except Exception:
        # Some classes may require n argument
        func = func_cls(2)
    bounds = func.bounds() if hasattr(func, 'bounds') else func_cls.bounds()
    test_points = get_test_points(bounds)
    for x_test in test_points:
        # NumPy evaluation
        f_numpy = func_cls.evaluate(x_test, xp=np)
        # Pyomo model and evaluation
        model = func.as_pyomo_model()
        for i, val in enumerate(x_test):
            model.x[i].value = val
        f_pyomo = pyo.value(model.obj.expr)
        assert np.allclose(f_numpy, f_pyomo, atol=1e-8), f"Mismatch for {func_cls.__name__} at {x_test}: numpy={f_numpy}, pyomo={f_pyomo}"
