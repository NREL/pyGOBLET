import inspect
import pytest
import pygoblet.problems.standard as bp
import numpy as np

# Get all classes defined in standard_problems.py (excluding base/template)
CLASSES = [
    obj for name, obj in inspect.getmembers(bp, inspect.isclass)
    if obj.__module__ == bp.__name__ and obj != bp.BenchmarkFunction
]

BACKENDS = ["NumPy", "CuPy", "PyTorch", "Dask", "JAX", "ndonnx", "sparse"]

def get_backend_namespace(backend_name):
    if backend_name == "NumPy":
        import numpy as np
        return np
    elif backend_name == "CuPy":
        import cupy as cp
        return cp
    elif backend_name == "PyTorch":
        import torch
        return torch
    elif backend_name == "Dask":
        import dask.array as da
        return da
    elif backend_name == "JAX":
        import jax.numpy as jnp
        return jnp
    elif backend_name == "ndonnx":
        import ndonnx
        return ndonnx
    elif backend_name == "sparse":
        import sparse
        return sparse
    else:
        raise ValueError(f"Unknown backend: {backend_name}")

# Check that the evaluate method at the argmin(s) returns the minimum value
@pytest.mark.parametrize("func_cls", CLASSES)
def test_function_min_at_argmin(func_cls):
    # Get min and argmin
    try:
        min_val = func_cls.min()
        argmin = func_cls.argmin()
    except Exception:
        # nD functions may require an instance to access min/argmin
        try:
            instance = func_cls(4)
            min_val = instance.min()
            argmin = instance.argmin()
        except Exception as e:
            pytest.skip(f"Skipping {func_cls.__name__}: {e}")
    if argmin is not None and min_val is not None:
        for xstar in argmin:
            try:
                f_val = func_cls.evaluate(np.array(xstar))
            except Exception as e:
                pytest.skip(f"Skipping evaluation for {func_cls.__name__} at {xstar}: {e}")
            assert np.allclose(f_val, min_val, atol=1e-4), f"{func_cls.__name__}: f(argmin)={f_val} != min={min_val} at {xstar}"

# Check that each backend works with the evaluate method
@pytest.mark.parametrize("backend", BACKENDS)
def test_backend_evaluate(backend):
    try:
        xp = get_backend_namespace(backend)
    except Exception as e:
        pytest.skip(f"Skipping {backend}: {e}")
    for func_cls in CLASSES:
        # Use argmin as test point if available
        dim = func_cls.DIM
        if isinstance(dim, int):
            prob = func_cls(dim)
        else:
            prob = func_cls(5)

        xstar = prob.argmin()
        if xstar is not None and len(xstar) > 0:
            xstar = xp.asarray(xstar[0])
            assert prob.evaluate(xstar) is not None, f"{func_cls.__name__} evaluate failed with {backend} backend"
        else:
            xstar = xp.ones(dim) if isinstance(dim, int) else xp.ones(5)
            assert prob.evaluate(xstar) is not None, f"{func_cls.__name__} evaluate failed with {backend} backend"
