.. Global Optimization Benchmarks documentation master file, created by
   sphinx-quickstart on Thu May 29 17:46:27 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

pyGOLD documentation
============================================

pyGOLD is a Python package for benchmarking global optimization algorithms. pyGOLD includes:

- A large collection of benchmark functions with various characteristics
- Support for both constrained and unconstrained optimization problems
- Pyomo model interface for algebraic modeling systems
- Tools for running and evaluating solvers on benchmark problems  
- COCOPP integration for postprocessing and visualization
- Custom postprocessing
- Example scripts demonstrating various use cases

Installation
------------
Install from PyPI:

.. code-block:: bash

   pip install pygold

Or from source:

.. code-block:: bash

   git clone <repository-url>
   cd <repo-directory>
   pip install -e .

Quick Start
-----------

The :mod:`pygold.benchmark_functions` module provides benchmark functions for testing solvers:

.. code-block:: python

   import pygold.benchmark_functions as bf
   
   # Create an Ackley function instance in 2D
   ackley = bf.Ackley(2)
   
   # Evaluate at a point
   x = [0.5, -0.3]
   result = ackley.evaluate(x)
   print(f"f({x}) = {result}")
   
   # Get problem information
   print(f"Minimum value: {ackley.min()}")
   print(f"Minimizer: {ackley.argmin()}")
   print(f"Bounds: {ackley.bounds()}")

The :mod:`pygold.benchmark_tools` module provides utilities for running and profiling solvers:

.. code-block:: python

   import scipy.optimize as opt
   import pygold.benchmark_functions as bf
   import pygold.benchmark_tools as bt
   import cocopp
   
   # Select problems and solvers
   problems = [bf.Ackley, bf.Rastrigin, bf.Griewank]
   solvers = [opt.shgo, opt.dual_annealing]
   
   # Run benchmark and generate COCO data
   bt.run_solvers(solvers, problems, test_dimensions=[2, 4, 6, 8, 10, 12], n_iters=5)
   
   # Configure and run COCOPP postprocessing
   bt.configure_testbed(problems, test_dimensions=[2, 4, 6, 8, 10, 12])
   cocopp.main(["output_data/shgo", "output_data/dual_annealing"])

Complete  examples can be found on the Github repo in ``Examples/``.

Function Classification
------------------------

Each benchmark function is tagged with one or more classification tags, which are used to organize and filter the available functions. The tags include:

- Unconstrained / Constrained: Whether the function has constraints
- Multimodal / Unimodal: Number of local/global minima
- Continuous / Discontinuous: Whether the function is continuous - functions with sharp ridges or drops are classified as discontinuous
- Differentiable / Non_differentiable: Whether the function is differentiable
- Separable / Non_separable: Whether the function can be separated into independent subproblems
- 1D, 2D, nD: Dimensionality of the function

You can access groups of functions by tag, e.g.:

.. code-block:: python

   import pygold.benchmark_functions as bf

   # All 2D functions
   problems = bf.__2D__

   # All multimodal functions
   problems = bf.__Multimodal__

   # All constrained functions
   problems = bf.__Constrained__

Contents
--------
.. toctree::
   :maxdepth: 2

   bench_functions
   bench_tools
   postprocessing
   examples

Indices
==================

* :ref:`genindex`
* :ref:`modindex`