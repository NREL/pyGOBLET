pyGOLD documentation
============================================

pyGOLD is a Python package for benchmarking global optimization algorithms. pyGOLD includes:

- A large collection of standard benchmark functions
- Benchmark functions inspired by real-world energy applications
- Tools for running solvers on benchmark problems
- Postprocessing tools for analyzing benchmark results
- Example scripts and tutorials demonstrating library usage

Installation
------------
Install from source:

.. code-block:: bash

   git clone https://github.nrel.gov/AI/pyGOLD.git
   cd pyGOLD
   pip install -e .


Available Problems
-------------------

PyGOLD includes standard benchmark problems and real-world inspired problems:

- Standard benchmark problems: :mod:`pygold.problems.standard`
- FLORIS wind farm optimization problems: :mod:`pygold.problems.floris` 

Quick Start
-----------

The :mod:`pygold.problems` module provides benchmark functions for testing solvers:

.. code-block:: python

   import pygold
   
   # Create an Ackley function instance in 2D
   ackley = pygold.problems.standard.Ackley(2)

   # Evaluate at a point
   x = [0.5, -0.3]
   result = ackley.evaluate(x)
   print(f"f({x}) = {result}")
   
   # Get problem information
   print(f"Minimum value: {ackley.min()}")
   print(f"Minimizer: {ackley.argmin()}")
   print(f"Bounds: {ackley.bounds()}")

A simple algorithm testing workflow:

.. code-block:: python

   import scipy.optimize as opt
   import pygold
   from pygold.optimizer import BaseOptimizer, OptimizationResult
   
   # Select test problems
   problems = pygold.get_standard_problems(["2D", "Unconstrained"])

   # Define solvers to benchmark
   class DualAnnealing(BaseOptimizer):
      deterministic = False
      n_points = 0
      def optimize(self, func, bounds, x0=None, constraints=None, **kwargs):
         result = opt.dual_annealing(func, bounds, **kwargs)
         return OptimizationResult(result.x, result.fun, algorithm="Dual Annealing")

   solvers = [DualAnnealing()]
   
   # Run benchmark and generate COCO data
   pygold.run_solvers(solvers, problems, test_dimensions=[2, 4, 5, 8, 10, 12], n_iters=5)
   
   # Run postprocessing
   pygold.postprocessing.postprocess_data(["output_data/DualAnnealing"], energy_file="output_data/energy_data.csv")

Complete examples and tutorials can be found on the Github repo in ``Examples/``.

.. toctree::
   :hidden:
   :maxdepth: 2

   problems
   
.. toctree::
   :hidden:
   :maxdepth: 1
   
   runners
   optimizer
   postprocessing
   examples
   genindex
