Standard Problems
==================

The standard problems module provides a collection of widely used benchmark functions for optimization algorithms, primarily from the `Virtual Library of Simulation Experiments <https://www.sfu.ca/~ssurjano/optimization.html>`_.

Each standard problem is tagged with one or more classification tags, which are used to organize and filter the available problems. The tags include:

- ``Unconstrained`` / ``Constrained``: Whether the function has constraints
- ``Multimodal`` / ``Unimodal``: Number of local/global minima
- ``Continuous`` / ``Discontinuous``: Whether the function is continuous - functions with sharp ridges or drops are classified as discontinuous
- ``Differentiable`` / ``Non_differentiable``: Whether the function is differentiable
- ``Separable`` / ``Non_separable``: Whether the function can be separated into independent subproblems
- ``1D``, ``2D``, ``nD``: Dimensionality of the function

The functions from VLSE also include the classification tags used in the original library:

- ``VLSE``
- ``Many_local_minima``
- ``Bowl_shaped``
- ``Plate_shaped``
- ``Valley_shaped``
- ``Steep_ridges_drops``
- ``Other``

You can access groups of functions by tag, e.g.:

.. code-block:: python

    # All 2D functions
    problems = bf.__2D__

    # All multimodal functions
    problems = bf.__Multimodal__

.. automodule:: pygold.problems.standard
    :members:
    :undoc-members: