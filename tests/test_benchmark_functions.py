import inspect
import global_opt_bench.benchmark_functions as bf

def test_benchmark_function_class_methods():
    # List of required method names
    required_methods = ['__init__', 'evaluate', 'min', 'bounds', 'argmin']

    # Get all classes defined in benchmark_functions.py
    classes = [
        obj for name, obj in inspect.getmembers(bf, inspect.isclass)
        if obj.__module__ == bf.__name__
    ]

    for cls in classes:
        for method in required_methods:
            assert hasattr(cls, method), f"{cls.__name__} missing method: {method}"
        # Check if the class has a docstring
        assert cls.__doc__, f"{cls.__name__} is missing a docstring"
