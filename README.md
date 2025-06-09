# pyGOLD: A Python Global Optimization Library

pyGOLD is a Python package for benchmarking global optimization algorithms.

## Features
- A large collection of benchmark functions (multimodal, unimodal, scalable, etc.)
- Tools for running and evaluating solvers on benchmark problems
- Performance ratio and performance profile computation
- Example files for running and comparing solvers

## Installation

Install from PyPI using pip.

```bash
pip install pygold
```

## Usage

### Accessing Benchmark Functions

The `pygold.benchmark_functions` module provides benchmark functions for testing solvers:

```python
import pygold.benchmark_functions as bf

# Access a specific function class directly
ackley = bf.Ackley(2)  # 2D Ackley function

# Evaluate the function at a point
x = np.array([0.0, 0.0])
value = ackley.evaluate(x)  # Function value at x

# Get bounds and known minimum
bounds = ackley.bounds()    # List of [lower, upper] for each dimension
min_val = ackley.min()      # Known minimum function value
argmin = ackley.argmin()    # Known minimizer(s)

# Access all available benchmark functions
all_functions = bf.__All__
```

### Benchmarking Tools

The `pygold.benchmark_tools` module provides utilities for running and profiling solvers:

```python
import pygold.benchmark_tools as bt

# Example: Run solvers on selected problems
solvers = [...]  # List of solver callables
problems = bf.__All__
results = bt.run_solvers_time(solvers, problems)

# Compute and plot performance profiles
ratios = bt.compute_performance_ratios(results)
profiles = bt.compute_performance_profiles(ratios)
bt.plot_performance_profiles(profiles)
```

## Function Classification

Each benchmark function is tagged with one or more classification tags, which are used to organize and filter the available functions. The tags include:

- `Unconstrained` / `Constrained`: Whether the function has constraints
- `Multimodal` / `Unimodal`: Number of local/global minima
- `Continuous` / `Discontinuous`: Whether the function is continuous - functions with sharp ridges or drops are classified as discontinuous
- `Differentiable` / `Non_differentiable`: Whether the function is differentiable
- `Separable` / `Non_separable`: Whether the function can be separated into independent subproblems
- `1D`, `2D`, `nD`: Dimensionality of the function

You can access groups of functions by tag, e.g.:

```python
# All 2D functions
problems = bf.__2D__

# All multimodal functions
problems = bf.__Multimodal__
```

## Documentation

Complete documentation is available at [LINK].

## Contributing

Contributions are welcome! Please open issues or pull requests for bug fixes, new features, or improvements.

## License

This project is licensed under the [LICENSE] License.