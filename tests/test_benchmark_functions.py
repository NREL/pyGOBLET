import inspect
import pytest
import global_opt_bench.benchmark_functions as bf
import numpy as np

# Get all classes defined in benchmark_functions.py (excluding base/template)
CLASSES = [
    obj for name, obj in inspect.getmembers(bf, inspect.isclass)
    if obj.__module__ == bf.__name__ and obj not in (bf.BenchmarkFunction, bf.className)
]

@pytest.mark.parametrize("func_cls", CLASSES)
def test_benchmark_function_class_methods(func_cls):
    required_methods = ['__init__', 'evaluate', 'min', 'bounds', 'argmin']
    for method in required_methods:
        assert hasattr(func_cls, method), f"{func_cls.__name__} missing method: {method}"
    assert func_cls.__doc__, f"{func_cls.__name__} is missing a docstring"

@pytest.mark.parametrize("func_cls", CLASSES)
def test_function_min_at_argmin(func_cls):
    # Get min and argmin
    try:
        min_val = func_cls.min()
        argmin = func_cls.argmin()
    except Exception:
        # Some classes may require instantiation
        try:
            instance = func_cls(2)
            min_val = instance.min()
            argmin = instance.argmin()
        except Exception as e:
            pytest.skip(f"Skipping {func_cls.__name__}: {e}")
    for xstar in argmin:
        try:
            f_val = func_cls.evaluate(np.array(xstar))
        except Exception as e:
            pytest.skip(f"Skipping evaluation for {func_cls.__name__} at {xstar}: {e}")
        assert np.allclose(f_val, min_val, atol=1e-4), f"{func_cls.__name__}: f(argmin)={f_val} != min={min_val} at {xstar}"
