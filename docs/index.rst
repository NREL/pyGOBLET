.. Global Optimization Benchmarks documentation master file, created by
   sphinx-quickstart on Thu May 29 17:46:27 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

pyGOLD documentation
============================================

pyGOLD is a Python package for benchmarking global optimization algorithms. pyGOLD includes:

- A large collection of standard benchmark functions
- Benchmark functions inspired by real-world energy applications
- Support for both constrained and unconstrained optimization problems
- Pyomo model interface for algebraic modeling systems
- Tools for running and evaluating solvers on benchmark problems
- COCOPP integration for postprocessing and visualization
- Custom postprocessing tools
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


Available Problems
-----------------

PyGOLD includes standard benchmark problems and real-world inspired problems:

- Standard benchmark problems: :mod:`pygold.problems.standard_problems`
- FLORIS wind farm optimization problems: :mod:`pygold.problems.floris_problems` 

Quick Start
-----------

The :mod:`pygold.problems` module provides benchmark functions for testing solvers:

.. code-block:: python

   from pygold.problems import standard_problems as bp
   
   # Create an Ackley function instance in 2D
   ackley = bp.Ackley(2)

   # Evaluate at a point
   x = [0.5, -0.3]
   result = ackley.evaluate(x)
   print(f"f({x}) = {result}")
   
   # Get problem information
   print(f"Minimum value: {ackley.min()}")
   print(f"Minimizer: {ackley.argmin()}")
   print(f"Bounds: {ackley.bounds()}")

A simple benchmark runner and postprocessing workflow:

.. code-block:: python

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

Complete examples can be found on the Github repo in ``Examples/``.

Contents
--------
.. toctree::
   :maxdepth: 2

   problems
   runners
   postprocessing
   examples

Indices
==================

* :ref:`genindex`
* :ref:`modindex`