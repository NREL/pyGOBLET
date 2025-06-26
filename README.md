# pyGOLD: A Python Global Optimization Library

pyGOLD is a Python package for benchmarking global optimization algorithms.

## Features

- A large collection of standard benchmark functions
- Benchmark functions inspired by real-world energy applications
- Support for both constrained and unconstrained optimization problems
- Pyomo model interface for algebraic modeling systems
- Tools for running and evaluating solvers on benchmark problems
- COCOPP integration for postprocessing and visualization
- Custom postprocessing tools
- Example scripts demonstrating various use cases

## Installation

Install from PyPI using pip.

```bash
pip install pygold
```

Or from source:

```bash
git clone <repository-url>
cd <repo-directory>
pip install -e .
```

## Available Problems

PyGOLD includes standard benchmark problems and real-world inspired problems:

- Standard benchmark problems: `pygold.problems.standard_problems`
- FLORIS wind farm optimization problems: `pygold.problems.floris_problems`

### Function Classification

Each standard benchmark function is tagged with one or more classification tags, which are used to organize and filter the available functions. The tags include:

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

## Usage

### Accessing Benchmark Functions

The `pygold.problems.standard_problems` module provides standard benchmark functions for testing solvers:

```python
from pygold.problems import standard_problems as bp

# Access a specific function class directly
ackley = bp.Ackley(2)  # 2D Ackley function

# Evaluate the function at a point
x = np.array([0.0, 0.0])
value = ackley.evaluate(x)  # Function value at x

# Get bounds and known minimum
bounds = ackley.bounds()    # List of [lower, upper] for each dimension
min_val = ackley.min()      # Known minimum function value
argmin = ackley.argmin()    # Known minimizer(s)

# Access all available benchmark functions
all_functions = bp.__All__
```

### Benchmarking a Solver

```python
import scipy.optimize as opt
import pygold
from pygold.problems import standard_problems as bp
import cocopp

# Select problems and solvers
problems = bp.__nD__ # All n-dimensional standard benchmark problems
solvers = [opt.shgo, opt.dual_annealing]

# Run benchmark and generate COCO data
pygold.run_solvers(solvers, problems, test_dimensions=[2, 4, 5, 8, 10, 12], n_iters=5)

# Configure and run COCOPP postprocessing
pygold.configure_testbed(problems, test_dimensions=[2, 4, 5, 8, 10, 12], n_solvers=2)
cocopp.main(["output_data/shgo", "output_data/dual_annealing"])
```

## Documentation

Complete documentation is available at [LINK].

## Contributing

Contributions are welcome! Please open issues or pull requests for bug fixes, new features, or improvements.

## License

This project is licensed under the [LICENSE] License.
