Postprocessing Tools
====================

PyGOLD provides tools for postprocessing benchmark results through two main approaches:

1. Performance profiles via the ``postprocessing`` module
2. COCOPP integration via the ``cocopp_interface`` module

See ``Examples/`` on the GitHub repo for example scripts demonstrating these features.

Performance Profiles
-------------------

The ``postprocessing`` module provides tools to generate performance profiles based on solver benchmark data.

.. automodule:: pygold.postprocessing
    :members:
    :undoc-members:

COCOPP Integration
-----------------

The ``cocopp_interface`` module provides integration with COCOPP (COmparing Continuous Optimizers) for postprocessing. Note that using COCOPP for postprocessing requires six test dimensions. To postprocess results from a single dimension, you can use the ``pygold.postprocessing`` module instead.

.. automodule:: pygold.cocopp_interface.interface
    :members:
    :undoc-members:
