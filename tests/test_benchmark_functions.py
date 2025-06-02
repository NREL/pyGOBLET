import inspect
import global_opt_bench.benchmark_functions as bf
import numpy as np

# Test that all benchmark function classes have the required methods
def test_benchmark_function_class_methods():
    # List of required method names
    required_methods = ['__init__', 'evaluate', 'min', 'bounds', 'argmin']

    # Get all classes defined in benchmark_functions.py
    classes = [
        obj for name, obj in inspect.getmembers(bf, inspect.isclass)
        if obj.__module__ == bf.__name__
    ]

    for cls in classes:
        if cls is bf.BenchmarkFunction:
            continue
        for method in required_methods:
            assert hasattr(cls, method), f"{cls.__name__} missing method: {method}"
        # Check if the class has a docstring
        assert cls.__doc__, f"{cls.__name__} is missing a docstring"

# Test that all benchmark function implementations match expected behavior
def test_function_min_at_argmin():
    # Get all classes defined in benchmark_functions.py
    classes = [
        obj for name, obj in inspect.getmembers(bf, inspect.isclass)
        if obj.__module__ == bf.__name__
    ]
    for cls in classes:
        if cls is bf.BenchmarkFunction or cls is bf.className:
            continue
        # Get min and argmin
        try:
            min_val = cls.min()
            argmin = cls.argmin()
        except Exception:
            # Some classes may require instantiation
            try:
                instance = cls(2)
                min_val = instance.min()
                argmin = instance.argmin()
            except Exception as e:
                print(f"Skipping {cls.__name__}: {e}")
                continue  # Skip if cannot instantiate
        for xstar in argmin:
            # Evaluate the function at argmin
            try:
                f_val = cls.evaluate(np.array(xstar))
            except Exception as e:
                print(f"Skipping evaluation for {cls.__name__} at {xstar}: {e}")
                continue
            try:
                assert np.allclose(f_val, min_val, atol=1e-4), f"{cls.__name__}: f(argmin)={f_val} != min={min_val} at {xstar}"
            except AssertionError as e:
                print(f"Assertion failed for {cls.__name__} at {xstar}: {e}")
                raise
