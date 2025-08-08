Adding Custom Test Problems
=============================

You can add your own test problems to PyGOLD by implementing a new problem class.

Basic Steps
-----------

1. **Subclass the base benchmark problem class**  
   
   Create a new class that inherits from ``pygold.problems.standard.BenchmarkProblem``.

2. **Define the required attributes and methods**  

   Implement:

   - ``evaluate(x)``: The objective function to be minimized or maximized.
   - ``bounds()``: A list of [lower, upper] bounds for each dimension.
   - ``min()``: The known minimum function value, if known. If unknown, return None.
   - ``argmin()``: A list of known optimum points, if known. If unknown, return None.
   - ``DIM``: Acceptable dimensions, either as an integer or a tuple. If using a tuple, -1 indicates no upper bound. (eg. (1, -1) for 1D to nD).

3. **Register your problem**  

   Add your new problem class to your workflow or contribute it to the PyGOLD problem registry.

Implementing these methods maintains compatibility with pyGOLD's runners and post-processing tools. Using array_api_compat for array operations is recommended to ensure compatibility with different array libraries (e.g., NumPy, PyTorch). See the source code for more examples.

Example
-------

.. code-block:: python

   from pygold.problems import BenchmarkProblem

   class Rosenbrock(BenchmarkFunction):
    """
    The Rosenbrock function is a n dimensional unimodal function.

    :Reference: https://www.sfu.ca/~ssurjano/rosen.html
    """

    # Acceptable dimensions. Either integer or tuple.
    # If tuple, use -1 to show 'no upper bound'.
    DIM = (1, -1)

    def __init__(self, n: int = DIM[0]) -> None:
        super().__init__(n)

    @staticmethod
    def evaluate(x, xp=None):
        """
        Evaluate the function at a given point.

        :param x: Input point (array-like)
        :param xp: Optional array API namespace (e.g., numpy, Torch)
        :return: Scalar function output
        """
        if xp is None:
            if isinstance(x, list):
                xp = np
            else:
                xp = array_api_compat.array_namespace(x)

        res = 0.0
        d = len(x)

        for i in range(d-1):
            res += 100 * (x[i + 1] - x[i] ** 2) ** 2 + (x[i] - 1) ** 2

        return res

    @staticmethod
    def min():
        """
        Returns known minimum function value.

        :return: Minimum value (float)
        """
        return 0.0

    def bounds(self):
        """
        Returns problem bounds.

        :return: List of [lower, upper] for each dimension
        """
        return [[-5, 10] for _ in range(self._ndims)]

    def argmin(self):
        """
        Returns function argmin.

        :return: List of minimizer(s)
        """
        return [[1.0 for i in range(self._ndims)]]

Tips
-----
- Functions should be implemented as staticmethods if possible to avoid unnecessary instance creation.
- For compatibility with runners and post-processing, ensure your problem class follows the interface of existing problems.
- You can implement additional methods or metadata as needed.
- To maintain pyomo compatibility, ensure your objective function is defined with a symbolic expression - not vectorized operations.