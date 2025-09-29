# pyGOBLET: A Python Global Optimization Benchmarking Library for Evaluation and Testing

pyGOBLET is a Python package for benchmarking global optimization algorithms.

## Features

- A large collection of standard benchmark functions
- Benchmark functions inspired by real-world energy applications
- Tools for running solvers on benchmark problems
- Postprocessing tools for analyzing benchmark results
- Example scripts and tutorials demonstrating library usage

## Installation

Install from source:

```bash
pip install pygoblet
```

## Available Problems

pyGOBLET includes standard benchmark problems and real-world inspired problems:

- Standard benchmark problems: `pygoblet.problems.standard`
- FLORIS wind farm optimization problems: `pygoblet.problems.floris`

### Function Classification

Each standard benchmark function is tagged with one or more classification tags, which are used to organize and filter the available functions. The tags include:

- `Unconstrained` / `Constrained`: Whether the function has constraints
- `Multimodal` / `Unimodal`: Number of local/global minima
- `Continuous` / `Discontinuous`: Whether the function is continuous - functions with sharp ridges or drops are classified as discontinuous
- `Differentiable` / `Non_differentiable`: Whether the function is differentiable
- `Separable` / `Non_separable`: Whether the function can be separated into independent subproblems
- `1D`, `2D`, `nD`: Dimensionality of the function

You can access groups of functions by tag using the ``get_standard_problems`` function:

```python
import pygoblet

# All 2D functions
problems = pygoblet.get_standard_problems(["2D"])

# All problems that are multimodal, unconstrained, and n-Dimensional
problems = pygoblet.get_standard_problems(["Multimodal", "Unconstrained", "nD"])
```

## Usage

### Accessing Benchmark Functions

The `pygoblet.problems.standard` module provides standard benchmark functions for testing solvers. For example:

```python
import pygoblet

# Create an Ackley function instance in 2D
ackley = pygoblet.problems.standard.Ackley(2)

# Evaluate at a point
x = [0.5, -0.3]
result = ackley.evaluate(x)
print(f"f({x}) = {result}")

# Get problem information
print(f"Minimum value: {ackley.min()}")
print(f"Minimizer: {ackley.argmin()}")
print(f"Bounds: {ackley.bounds()}")
```

### Benchmarking a Solver

```python
import scipy.optimize as opt
import pygoblet
from pygoblet.optimizer import BaseOptimizer, OptimizationResult

# Select test problems
problems = pygoblet.get_standard_problems(["2D", "Unconstrained"])

# Define solver to benchmark
class DualAnnealing(BaseOptimizer):
    deterministic = False
    n_points = 0
    def optimize(self, func, bounds, x0=None, constraints=None, **kwargs):
        result = opt.dual_annealing(func, bounds, **kwargs)
        return OptimizationResult(result.x, result.fun, algorithm="Dual Annealing")

solvers = [DualAnnealing()]

# Run benchmark and generate COCO data
pygoblet.run_solvers(solvers, problems, test_dimensions=[2, 4, 5, 8, 10, 12], n_iters=5)

# Run postprocessing
pygoblet.postprocessing.postprocess_data(["output_data/DualAnnealing"], energy_file="output_data/energy_data.csv")
```

## Documentation

Complete documentation is available [here](https://nrel.github.io/pyGOBLET/).

## Contributing

Contributions are welcome! Please open issues or pull requests for bug fixes, new features, or improvements.

## License

This project is licensed under the BSD-3-Clause License.

---

_NREL Software Record number: SWR-25-118_
