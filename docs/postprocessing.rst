Postprocessing Tools
====================

PyGOLD provides tools for postprocessing benchmark results through two main approaches:

1. COCOPP integration via the ``cocopp_interface`` module
2. Custom postprocessing via the ``postprocessing`` module

See ``Examples/`` on the GitHub repo for example scripts demonstrating these features.

COCOPP Integration
-------------------

The ``cocopp_interface`` module provides integration with COCOPP (COmparing Continuous Optimizers) for postprocessing. Note that using COCOPP for postprocessing requires six test dimensions. To postprocess results from a single dimension, you can use the ``pygold.postprocessing`` module instead.

.. automodule:: pygold.cocopp_interface.interface
    :members:
    :undoc-members:

Custom Postprocessing
----------------------

The ``postprocessing`` module provides tools to generate performance statistics based on solver benchmark data. This includes postprocessing for function evaluations and energy statistics.

.. automodule:: pygold.postprocessing
    :members:
    :undoc-members:
