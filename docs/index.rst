.. Global Optimization Benchmarks documentation master file, created by
   sphinx-quickstart on Thu May 29 17:46:27 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

pyGOLD documentation
============================================

pyGOLD is a Python package for benchmarking global optimization algorithms. pyGOLD includes:

- A large collection of benchmark functions
- Tools for running and evaluating solvers on benchmark problems
- Performance ratio and performance profile computation
- Example files for running and comparing solvers

Installation
------------
.. code-block:: bash

   **Installation code**

Usage
-----
Example usage:

.. code-block:: python

   import pygold.benchmark_functions as bf
   import pygold.benchmark_tools as bt
   # Make a simple example

Contents
-------------
.. toctree::
   :maxdepth: 2

   bench_functions
   bench_tools
   examples

Indices
==================

* :ref:`genindex`
* :ref:`modindex`