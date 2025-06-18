Postprocessing Tools
====================

The postprocessing module provides integration with COCOPP (COmparing Continuous Optimizers) for generating performance plots and tables.

Custom Testbed Configuration
----------------------------

The CustomTestbed class extends COCOPP's functionality to work with pyGOLD benchmark functions.

Key Features:

- **Configurable dimensions**: Specify test dimensions that match your experimental setup
- **Custom function names**: Use meaningful names for your benchmark problems  
- **Function grouping**: Organize functions into logical categories for analysis

Usage with Benchmark Tools
---------------------------

The postprocessing module is typically used in conjunction with :mod:`pygold.benchmark_tools`:

.. code-block:: python

   import pygold.benchmark_tools as bt
   import pygold.benchmark_functions as bf
   
   # Configure testbed for your problems
   problems = [bf.Ackley, bf.Rastrigin, bf.Griewank]
   bt.configure_testbed(problems, test_dimensions=[2, 4, 6, 8, 10, 12])
   
   # The testbed is now configured for COCOPP processing

Once configured, the custom testbed works seamlessly with standard COCOPP commands. Note that using less than 6 test dimensions will result in repeated graphs in the output.

.. code-block:: python

   import cocopp
   
   # After running bt.configure_testbed() and bt.run_solvers()
   cocopp.main(["output_data/solver1", "output_data/solver2"])

This generates a HTML report with:

- Performance profiles across different targets
- Empirical cumulative distribution functions (ECDFs) 
- Scaling behavior with problem dimension
- Statistical data tables
