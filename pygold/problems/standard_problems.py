import numpy as np
import array_api_compat
try:
    import pyomo.environ as pyo
except ImportError:
    pyo = None
import sys

# Function categories

# General function categories
__All__ = []
__Multimodal__ = []
__Unimodal__ = []
__Continuous__ = []
__Discontinuous__ = []
__Differentiable__ = []
__Non_differentiable__ = []
__Separable__ = []
__Non_separable__ = []
__Unconstrained__ = []
__Constrained__ = []

# VLSE function categories
__VLSE__ = []
__Many_local_minima__ = []
__Bowl_shaped__ = []
__Plate_shaped__ = []
__Valley_shaped__ = []
__Steep_ridges_drops__ = []
__Other__ = []

# Dimension categories
__nD__ = []
__4D__ = []
__2D__ = []
__1D__ = []

def _get_abs(xp):
    """Helper function to get the appropriate absolute value function."""
    if pyo is not None and xp == pyo:
        import pyomo.core.expr as expr
        return expr.numvalue.NumericValue.__abs__
    else:
        return xp.abs

def tag(tags):
    """Decorator to register classes into categories."""
    def decorator(cls):
        mod = sys.modules[__name__]
        if not hasattr(mod, "__All__"):
            mod.__All__ = []
        mod.__All__.append(cls)
        for t in tags:
            list_name = f"__{t}__"
            if not hasattr(mod, list_name):
                setattr(mod, list_name, [])
            getattr(mod, list_name).append(cls)
        return cls
    return decorator

class BenchmarkFunction:
    """
    Superclass for benchmark functions. Provides a Pyomo model interface.
    """

    def __init__(self, n: int):
        self._ndims = n

    def as_pyomo_model(self):
        """
        Returns a Pyomo ConcreteModel for this benchmark function.

        :return: Pyomo ConcreteModel with variables and objective.
        """
        if pyo is None:
            raise ImportError("Pyomo is not installed. Please install pyomo to use this feature.")
        model = pyo.ConcreteModel()
        n = self._ndims
        bounds = self.bounds()
        model.x = pyo.Var(range(n), domain=pyo.Reals)

        # Set variable bounds
        for i in range(n):
            lb, ub = bounds[i]
            model.x[i].setlb(lb)
            model.x[i].setub(ub)
            if lb == -float('inf') and ub != float('inf'):
                model.x[i].value = ub / 2
            elif ub == float('inf') and lb != -float('inf'):
                model.x[i].value = lb * 2
            elif lb == -float('inf') and ub == float('inf'):
                model.x[i].value = 1e-3
            else:
                model.x[i].value = (lb + ub) / 2

            # Ensure no variable is initialized to zero if it has bounds
            # This prevents issues with functions with no derivative at zero
            if model.x[i].value == 0:
                model.x[i].value = (lb + 1.0001 * ub) / 2

        # Use symbolic Pyomo expression if available
        try:
            expr = self.evaluate(model.x, xp=pyo)
        except Exception:
            raise NotImplementedError("This benchmark function does not support symbolic Pyomo expressions.")
        model.obj = pyo.Objective(expr=expr, sense=pyo.minimize)

        # Add constraints if they exist
        if hasattr(self, 'constraints'):
            model.constraints = pyo.ConstraintList()
            for c in self.constraints():
                try:
                    const = (lambda x, xp=pyo: c(x, xp=xp) >= 0)(model.x, xp=pyo)
                except Exception:
                    raise NotImplementedError("This benchmark function does not support symbolic Pyomo constraints.")
                model.constraints.add(expr=const)
        return model

    def constraints(self):
        """
        Returns a list of constraint functions for this benchmark.
        Each function should take (x, xp=None) and return a scalar (<= 0 when
        satisfied), or a Pyomo relational expression if used with Pyomo.
        By default, returns an empty list (no constraints).
        """
        return []

# =============================================================================
# Virtual Library of Simulation Experiments (VLSE) Test Problems
# Link: https://www.sfu.ca/~ssurjano/optimization.html
# =============================================================================
@tag(["VLSE", "Many_local_minima", "Unconstrained", "Multimodal", "Continuous", "nD", "Differentiable", "Non_separable"])
class Ackley(BenchmarkFunction):
    """
    The Ackley function is a N-dimensional function with many local minima
    throughout the domain.

    :References: https://www.sfu.ca/~ssurjano/ackley.html
    """

    # Acceptable dimensions. Either integer or tuple.
    # If tuple, use -1 to show 'no upper bound'.
    DIM = (1, -1)

    def __init__(self, n: int = DIM[0]) -> None:
        super().__init__(n)

    @staticmethod
    def evaluate(x, a=20.0, b=0.2, c=2*np.pi, xp=None):
        """
        Evaluate the Ackley function.

        :param x: n-d input point (array-like)
        :param a: float, default 20
        :param b: float, default 0.2
        :param c: float, default 2π
        :param xp: Optional array API namespace (e.g., numpy, Torch)
        :return: Scalar function output
        """
        if xp is None:
            if isinstance(x, list):
                xp = np
            else:
                xp = array_api_compat.array_namespace(x)

        n = len(x)
        term1 = -a * xp.exp(-b * xp.sqrt((1/n) * sum(x[i]**2 for i in range(n))))
        term2 = -xp.exp((1/n) * sum(xp.cos(c * x[i]) for i in range(n)))

        res = term1 + term2 + a + xp.exp(1)

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
        return [[-32.768, 32.768] for i in range(self._ndims)]

    def argmin(self):
        """
        Returns function argmin.

        :return: List of minimizer(s)
        """
        return [[0.0 for i in range(self._ndims)]]

@tag(["VLSE", "Many_local_minima", "Unconstrained", "Multimodal", "2D", "Continuous", "Non_differentiable", "Non_separable"])
class Bukin6(BenchmarkFunction):
    """
    Bukin Function N. 6 is a 2D function with many local minima along a ridge.

    :References: https://www.sfu.ca/~ssurjano/bukin6.html
    """

    # Acceptable dimensions. Either integer or tuple.
    # If tuple, use -1 to show 'no upper bound'.
    DIM = 2

    def __init__(self, n: int = 2) -> None:
        super().__init__(n)

    @staticmethod
    def evaluate(x, xp=None):
        """
        Evaluate the Bukin6 function.

        :param x: 2D input point (array-like)
        :param xp: Optional array API namespace (e.g., numpy, Torch)
        :return: Scalar function output
        """
        if xp is None:
            if isinstance(x, list):
                xp = np
            else:
                xp = array_api_compat.array_namespace(x)
        abs_fn = _get_abs(xp)

        x1 = x[0]
        x2 = x[1]
        term1 = 100 * xp.sqrt(abs_fn(x2 - 0.01 * x1**2))
        term2 = 0.01 * abs_fn(x1 + 10)
        return term1 + term2

    @staticmethod
    def min():
        """
        Returns known minimum function value.

        :return: Minimum value (float)
        """
        return 0.0

    @staticmethod
    def bounds():
        """
        Returns problem bounds.

        :return: List of [lower, upper] for each dimension
        """
        return [[-15, -5], [-3, 3]]

    @staticmethod
    def argmin():
        """
        Returns function argmin.

        :return: List of minimizer(s)
        """
        return [[-10, 1]]

@tag(["VLSE", "Many_local_minima", "Unconstrained", "Multimodal", "2D", "Continuous", "Non_separable"])
class CrossInTray(BenchmarkFunction):
    """
    The Cross-in-Tray is a 2D function with many local minima and
    four global minima.

    :References: http://infinity77.net/global_optimization/test_functions_nd_C.html
    """

    # Acceptable dimensions. Either integer or tuple.
    # If tuple, use -1 to show 'no upper bound'.
    DIM = 2

    def __init__(self, n: int = 2) -> None:
        super().__init__(n)

    @staticmethod
    def evaluate(x, xp=None):
        """
        Evaluate the Cross-in-Tray function.

        :param x: 2D input point (array-like)
        :param xp: Optional array API namespace (e.g., numpy, Torch)
        :return: Scalar function output
        """
        # Unpack the input vector
        x1 = x[0]
        x2 = x[1]

        if xp is None:
            if isinstance(x, list):
                xp = np
            else:
                xp = array_api_compat.array_namespace(x)
        abs_fn = _get_abs(xp)

        # Compute the Cross-in-Tray function
        term1 = abs_fn(xp.sin(x1) * xp.sin(x2))
        term2 = xp.exp(abs_fn(100 - xp.sqrt(x1**2 + x2**2) / np.pi))

        res = -0.0001 * (term1 * term2 + 1)**0.1

        return res

    @staticmethod
    def min():
        """
        Returns known minimum function value.

        :return: Minimum value (float)
        """
        return -2.062611870822739

    @staticmethod
    def bounds():
        """
        Returns problem bounds.

        :return: List of [lower, upper] for each dimension
        """
        return [[-10, 10], [-10, 10]]

    @staticmethod
    def argmin():
        """
        Returns function argmin.

        :return: List of minimizer(s)
        """
        return [[1.349406608602084, -1.349406608602084], [1.349406608602084, 1.349406608602084], [-1.349406608602084, 1.349406608602084], [-1.349406608602084, -1.349406608602084]]

@tag(["VLSE", "Many_local_minima", "Unconstrained", "Multimodal", "2D", "Continuous"])
class DropWave(BenchmarkFunction):
    """
    The Drop-Wave function is a multimodal 2D function with many local minima
    and one global minimum.

    :References: https://www.sfu.ca/~ssurjano/drop.html
    """

    # Acceptable dimensions. Either integer or tuple.
    # If tuple, use -1 to show 'no upper bound'.
    DIM = 2

    def __init__(self, n: int = 2) -> None:
        super().__init__(n)

    @staticmethod
    def evaluate(x, xp=None):
        """
        Evaluate the Drop-Wave function.

        :param x: 2D input point (array-like)
        :param xp: Optional array API namespace (e.g., numpy, Torch)
        :return: Scalar function output
        """
        # Unpack the input vector
        x1 = x[0]
        x2 = x[1]

        if xp is None:
            if isinstance(x, list):
                xp = np
            else:
                xp = array_api_compat.array_namespace(x)

        # Compute the Drop-Wave function
        num = 1 + xp.cos(12 * xp.sqrt(x1**2 + x2**2))
        denom = 0.5 * (x1**2 + x2**2) + 2

        res = - num / denom

        return res

    @staticmethod
    def min():
        """
        Returns known minimum function value.

        :return: Minimum value (float)
        """
        return -1.0

    @staticmethod
    def bounds():
        """
        Returns problem bounds.

        :return: List of [lower, upper] for each dimension
        """
        return [[-5.12, 5.12], [-5.12, 5.12]]

    @staticmethod
    def argmin():
        """
        Returns function argmin.

        :return: List of minimizer(s)
        """
        return [[0.0, 0.0]]

@tag(["VLSE", "Many_local_minima", "Unconstrained", "Multimodal", "2D", "Continuous", "Differentiable", "Non_separable"])
class EggHolder(BenchmarkFunction):
    """
    The Eggholder function is a 2D function with many local minima and
    one global minimum.

    :References: http://infinity77.net/global_optimization/test_functions_nd_E.html
    """

    # Acceptable dimensions. Either integer or tuple.
    # If tuple, use -1 to show 'no upper bound'.
    DIM = 2

    def __init__(self, n: int = 2) -> None:
        super().__init__(n)

    @staticmethod
    def evaluate(x, xp=None):
        """
        Evaluate the Eggholder function.

        :param x: 2D input point (array-like)
        :param xp: Optional array API namespace (e.g., numpy, Torch)
        :return: Scalar function output
        """
        # Unpack the input vector
        x1 = x[0]
        x2 = x[1]

        if xp is None:
            if isinstance(x, list):
                xp = np
            else:
                xp = array_api_compat.array_namespace(x)
        abs_fn = _get_abs(xp)

        # Compute the Eggholder function
        term1 = -(x2 + 47) * xp.sin(xp.sqrt(abs_fn(x1 / 2 + x2 + 47)))
        term2 = -x1 * xp.sin(xp.sqrt(abs_fn(x1 - (x2 + 47))))

        res = term1 + term2

        return res

    @staticmethod
    def min():
        """
        Returns known minimum function value.

        :return: Minimum value (float)
        """
        return -959.640662711

    @staticmethod
    def bounds():
        """
        Returns problem bounds.

        :return: List of [lower, upper] for each dimension
        """
        return [[-512, 512], [-512, 512]]

    @staticmethod
    def argmin():
        """
        Returns function argmin.

        :return: List of minimizer(s)
        """
        return [[512, 404.2319]]

@tag(["VLSE", "Many_local_minima", "Unconstrained", "Multimodal", "1D", "Continuous", "Differentiable"])
class GramacyLee(BenchmarkFunction):
    """
    The Gramacy-Lee function is a 1D function with many local minima
    and one global minimum.

    :References: https://www.sfu.ca/~ssurjano/grlee12.html
    """

    # Acceptable dimensions. Either integer or tuple.
    # If tuple, use -1 to show 'no upper bound'.
    DIM = 1

    def __init__(self, n: int = 1) -> None:
        super().__init__(n)

    @staticmethod
    def evaluate(x, xp=None):
        """
        Evaluate the Gramacy-Lee function.

        :param x: 1D input point
        :param xp: Optional array API namespace (e.g., numpy, Torch)
        :return: Scalar function output
        """
        if xp is not None:
            x = x[0]
        if xp is None:
            if isinstance(x, list):
                xp = np
            else:
                xp = array_api_compat.array_namespace(x)

        # Compute the Gramacy-Lee function
        term1 = xp.sin(10 * np.pi * x) / (2 * x)
        term2 = (x - 1)**4

        res = term1 + term2

        return res

    @staticmethod
    def min():
        """
        Returns known minimum function value.

        :return: Minimum value (float)
        """
        return -0.869011134989500

    @staticmethod
    def bounds():
        """
        Returns problem bounds.

        :return: List of [lower, upper] for each dimension
        """
        return [[0.5, 2.5]]

    @staticmethod
    def argmin():
        """
        Returns function argmin.

        :return: Minimizer
        """
        return [0.548563444114526]

@tag(["VLSE", "Many_local_minima", "Unconstrained", "Multimodal", "nD", "Continuous", "Differentiable", "Non_separable"])
class Griewank(BenchmarkFunction):
    """
    The Griewank function is a N-dimensional function with many local minima
    and one global minimum.

    :References: https://www.sfu.ca/~ssurjano/griewank.html
    """

    # Acceptable dimensions. Either integer or tuple.
    # If tuple, use -1 to show 'no upper bound'.
    DIM = (1, -1)

    def __init__(self, n: int = DIM[0]) -> None:
        super().__init__(n)

    @staticmethod
    def evaluate(x, xp=None):
        """
        Evaluate the Griewank function.

        :param x: n-d input point (array-like)
        :param xp: Optional array API namespace (e.g., numpy, Torch)
        :return: Scalar function output
        """
        if xp is None:
            if isinstance(x, list):
                xp = np
            else:
                xp = array_api_compat.array_namespace(x)
        n = len(x)
        indices = range(1, n + 1)
        sum_term = sum(x[i]**2 for i in range(n)) / 4000
        prod_term = 1
        for i in range(n):
            prod_term *= xp.cos(x[i] / xp.sqrt(indices[i]))
        return 1 + sum_term - prod_term

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
        return [[-600, 600] for i in range(self._ndims)]

    def argmin(self):
        """
        Returns function argmin.

        :return: List of minimizer(s)
        """
        return [[0.0 for i in range(self._ndims)]]

@tag(["VLSE", "Many_local_minima", "Unconstrained", "Multimodal", "2D", "Continuous", "Differentiable", "Separable"])
class HolderTable(BenchmarkFunction):
    """
    The Holder Table function is a 2D function with many local minima
    and four global minima.

    :References: https://www.sfu.ca/~ssurjano/holder.html
    """

    # Acceptable dimensions. Either integer or tuple.
    # If tuple, use -1 to show 'no upper bound'.
    DIM = 2

    def __init__(self, n: int = 2) -> None:
        super().__init__(n)

    @staticmethod
    def evaluate(x, xp=None):
        """
        Evaluate the Holder Table function.

        :param x: 2D input point (array-like)
        :param xp: Optional array API namespace (e.g., numpy, Torch)
        :return: Scalar function output
        """
        # Unpack the input vector
        x1 = x[0]
        x2 = x[1]

        if xp is None:
            if isinstance(x, list):
                xp = np
            else:
                xp = array_api_compat.array_namespace(x)

        abs_fn = _get_abs(xp)

        # Compute the Holder Table function
        term1 = xp.sin(x1) * xp.cos(x2)
        term2 = xp.exp(abs_fn(1 - (xp.sqrt(x1**2 + x2**2) / np.pi)))

        res = -abs_fn(term1 * term2)

        return res

    @staticmethod
    def min():
        """
        Returns known minimum function value.

        :return: Minimum value (float)
        """
        return -19.2085

    @staticmethod
    def bounds():
        """
        Returns problem bounds.

        :return: List of [lower, upper] for each dimension
        """
        return [[-10.0, 10.0], [-10.0, 10.0]]

    @staticmethod
    def argmin():
        """
        Returns function argmin.

        :return: List of minimizer(s)
        """
        return [[8.05502, 9.66459], [-8.05502, -9.66459], [8.05502, -9.66459], [-8.05502, 9.66459]]

@tag(["VLSE", "Many_local_minima", "Unconstrained", "Multimodal", "2D", "Continuous", "Differentiable", "Non_separable"])
class Langermann(BenchmarkFunction):
    """
    The Langermann function is a 2D function with many local minima and
    one global minimum.

    :References:
        https://www.sfu.ca/~ssurjano/langer.html
        https://infinity77.net/global_optimization/test_functions_nd_L.html#go_benchmark.Langermann
    """

    # Acceptable dimensions. Either integer or tuple.
    # If tuple, use -1 to show 'no upper bound'.
    DIM = 2

    def __init__(self, n: int = 2) -> None:
        super().__init__(n)

    @staticmethod
    def evaluate(x, xp=None):
        """
        Evaluate the Langermann function.

        :param x: 2D input point (array-like)
        :param xp: Optional array API namespace (e.g., numpy, Torch)
        :return: Scalar function output
        """
        if xp is None:
            if isinstance(x, list):
                xp = np
            else:
                xp = array_api_compat.array_namespace(x)

        A = [[3, 5], [5, 2], [2, 1], [1, 4], [7, 9]]
        c = [1, 2, 5, 2, 3]

        res = 0
        for i in range(5):
            inner = 0
            for j in range(2):
                inner += (x[j] - A[i][j]) ** 2
            res += c[i] * xp.exp(-inner / np.pi) * xp.cos(np.pi * inner)
        return -res

    @staticmethod
    def min():
        """
        Returns known minimum function value.

        :return: Minimum value (float)
        """
        return -5.1621259

    @staticmethod
    def bounds():
        """
        Returns problem bounds.

        :return: List of [lower, upper] for each dimension
        """
        return [[0.0, 10.0], [0.0, 10.0]]

    @staticmethod
    def argmin():
        """
        Returns function argmin.

        :return: List of minimizer(s)
        """
        return [[2.002992, 1.006096]]

@tag(["VLSE", "Many_local_minima", "Unconstrained", "Multimodal", "nD", "Continuous"])
class Levy(BenchmarkFunction):
    """
    The Levy Function is a N-dimensional function with many local minima and
    one global minimum.

    :References: https://www.sfu.ca/~ssurjano/levy.html
    """

    # Acceptable dimensions. Either integer or tuple.
    # If tuple, use -1 to show 'no upper bound'.
    DIM = (1, -1)

    def __init__(self, n: int = DIM[0]) -> None:
        super().__init__(n)

    @staticmethod
    def evaluate(x, xp=None):
        """
        Evaluate the Levy function.

        :param x: n-d input point (array-like)
        :param xp: Optional array API namespace (e.g., numpy, Torch)
        :return: Scalar function output
        """
        if xp is None:
            if isinstance(x, list):
                xp = np
            else:
                xp = array_api_compat.array_namespace(x)

        if hasattr(xp, "asarray"):
            x = xp.asarray(x)
            w = 1 + (x - 1) / 4
            term1 = xp.sin(np.pi * w[0])**2
            term2 = xp.sum((w[:-1] - 1)**2 * (1 + 10 * xp.sin(np.pi * w[:-1] + 1)**2))
            term3 = (w[-1] - 1)**2 * (1 + xp.sin(2 * np.pi * w[-1])**2)
        else:
            w = [1 + (x[i] - 1) / 4 for i in range(len(x))]
            term1 = xp.sin(np.pi * w[0])**2
            term2 = sum((w[i] - 1)**2 * (1 + 10 * xp.sin(np.pi * w[i] + 1)**2) for i in range(len(w) - 1))
            term3 = (w[-1] - 1)**2 * (1 + xp.sin(2 * np.pi * w[-1])**2)

        res = term1 + term2 + term3

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
        return [[-10.0, 10.0] for i in range(self._ndims)]

    def argmin(self):
        """
        Returns function argmin.

        :return: List of minimizer(s)
        """
        return [[1.0 for i in range(self._ndims)]]

@tag(["VLSE", "Many_local_minima", "Unconstrained", "Multimodal", "2D", "Continuous"])
class Levy13(BenchmarkFunction):
    """
    Levy 13 is a 2D function with many local minima and one global minimum.

    :References: https://www.sfu.ca/~ssurjano/levy13.html
    """

    # Acceptable dimensions. Either integer or tuple.
    # If tuple, use -1 to show 'no upper bound'.
    DIM = 2

    def __init__(self, n: int = 2) -> None:
        super().__init__(n)

    @staticmethod
    def evaluate(x, xp=None):
        """
        Evaluate the Levy 13 function.

        :param x: 2D input point (array-like)
        :param xp: Optional array API namespace (e.g., numpy, Torch)
        :return: Scalar function output
        """
        # Unpack the input vector
        x1 = x[0]
        x2 = x[1]

        if xp is None:
            if isinstance(x, list):
                xp = np
            else:
                xp = array_api_compat.array_namespace(x)

        # Compute the Levy function
        term1 = xp.sin(3 * np.pi * x1)**2
        term2 = (x1 - 1)**2 * (1 + xp.sin(3 * np.pi * x2)**2)
        term3 = (x2 - 1)**2 * (1 + xp.sin(2 * np.pi * x2)**2)

        res = term1 + term2 + term3

        return res

    @staticmethod
    def min():
        """
        Returns known minimum function value.

        :return: Minimum value (float)
        """
        return 0.0

    @staticmethod
    def bounds():
        """
        Returns problem bounds.

        :return: List of [lower, upper] for each dimension
        """
        return [[-10.0, 10.0], [-10.0, 10.0]]

    @staticmethod
    def argmin():
        """
        Returns function argmin.

        :return: List of minimizer(s)
        """
        return [[1.0, 1.0]]

@tag(["VLSE", "Many_local_minima", "Unconstrained", "Multimodal", "nD", "Continuous"])
class Rastrigin(BenchmarkFunction):
    """
    The Rastrigin function is a N-dimensional function with many local minima
    and one global minimum.

    :References: https://www.sfu.ca/~ssurjano/rastr.html
    """

    # Acceptable dimensions. Either integer or tuple.
    # If tuple, use -1 to show 'no upper bound'.
    DIM = (1, -1)

    def __init__(self, n: int = DIM[0]) -> None:
        super().__init__(n)

    @staticmethod
    def evaluate(x, xp=None):
        """
        Evaluate the Rastrigin function.

        :param x: n-d input point (array-like)
        :param xp: Optional array API namespace (e.g., numpy, Torch)
        :return: Scalar function output
        """
        if xp is None:
            if isinstance(x, list):
                xp = np
            else:
                xp = array_api_compat.array_namespace(x)

        d = len(x)
        term1 = 10 * d
        if hasattr(xp, "asarray"):
            x = xp.asarray(x)
            term2 = xp.sum(x**2 - 10 * xp.cos(2 * np.pi * x))
        else:
            term2 = sum(x[i]**2 - 10 * xp.cos(2 * np.pi * x[i]) for i in range(d))

        # Compute the Rastrigin function
        res = term1 + term2

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
        return [[-5.12, 5.12] for i in range(self._ndims)]

    def argmin(self):
        """
        Returns function argmin.

        :return: List of minimizer(s)
        """
        return [[0.0 for i in range(self._ndims)]]

@tag(["VLSE", "Many_local_minima", "Unconstrained", "Multimodal", "2D", "Continuous"])
class Schaffer2(BenchmarkFunction):
    """
    The second Schaffer function is a 2D function with many local minima and
    one global minimum.

    :References: https://www.sfu.ca/~ssurjano/schaffer2.html
    """

    # Acceptable dimensions. Either integer or tuple.
    # If tuple, use -1 to show 'no upper bound'.
    DIM = 2

    def __init__(self, n: int = 2) -> None:
        super().__init__(n)

    @staticmethod
    def evaluate(x, xp=None):
        """
        Evaluate the Schaffer 2 function.

        :param x: 2D input point (array-like)
        :param xp: Optional array API namespace (e.g., numpy, Torch)
        :return: Scalar function output
        """
        # Unpack the input vector
        x1 = x[0]
        x2 = x[1]

        if xp is None:
            if isinstance(x, list):
                xp = np
            else:
                xp = array_api_compat.array_namespace(x)

        # Compute the Schaffer function
        numer = xp.sin(x1**2 - x2**2)**2 - 0.5
        denom = (1 + 0.001 * (x1**2 + x2**2))**2

        res = 0.5 + numer / denom

        return res

    @staticmethod
    def min():
        """
        Returns known minimum function value.

        :return: Minimum value (float)
        """
        return 0.0

    @staticmethod
    def bounds():
        """
        Returns problem bounds.

        :return: List of [lower, upper] for each dimension
        """
        return [[-100.0, 100.0], [-100.0, 100.0]]

    @staticmethod
    def argmin():
        """
        Returns function argmin.

        :return: List of minimizer(s)
        """
        return [[0.0, 0.0]]

@tag(["VLSE", "Many_local_minima", "Unconstrained", "Multimodal", "2D", "Continuous"])
class Schaffer4(BenchmarkFunction):
    """
    The fourth Schaffer function is a 2D function with many local minima and
    four global minima.

    :References:
        https://www.sfu.ca/~ssurjano/schaffer4.html
        https://en.wikipedia.org/wiki/Test_functions_for_optimization
        Mishra, S. Some new test functions for global optimization and
        performance of repulsive particle swarm method. Munich Personal
        RePEc Archive, 2006, 2718
    """

    # Acceptable dimensions. Either integer or tuple.
    # If tuple, use -1 to show 'no upper bound'.
    DIM = 2

    def __init__(self, n: int = 2) -> None:
        super().__init__(n)

    @staticmethod
    def evaluate(x, xp=None):
        """
        Evaluate the Schaffer 4 function.

        :param x: 2D input point (array-like)
        :param xp: Optional array API namespace (e.g., numpy, Torch)
        :return: Scalar function output
        """
        # Unpack the input vector
        x1 = x[0]
        x2 = x[1]

        if xp is None:
            if isinstance(x, list):
                xp = np
            else:
                xp = array_api_compat.array_namespace(x)

        abs_fn = _get_abs(xp)

        # Compute the Schaffer function
        numer = xp.cos(xp.sin(abs_fn(x1**2 - x2**2)))**2 - 0.5
        denom = (1 + 0.001 * (x1**2 + x2**2))**2

        res = 0.5 + numer / denom

        return res

    @staticmethod
    def min():
        """
        Returns known minimum function value.

        :return: Minimum value (float)
        """
        return 0.292579

    @staticmethod
    def bounds():
        """
        Returns problem bounds.

        :return: List of [lower, upper] for each dimension
        """
        return [[-100.0, 100.0], [-100.0, 100.0]]

    @staticmethod
    def argmin():
        """
        Returns function argmin.

        :return: List of minimizer(s)
        """
        return [[0.0, 1.253115], [0.0, -1.253115], [1.253115, 0.0], [-1.253115, 0.0]]

@tag(["VLSE", "Many_local_minima", "Unconstrained", "Multimodal", "nD", "Continuous", "Differentiable", "Separable"])
class Schwefel(BenchmarkFunction):
    """
    The Schwefel function is a N-dimensional function with many local minima and
    one global minimum.

    :References: https://www.sfu.ca/~ssurjano/schwef.html
    """

    # Acceptable dimensions. Either integer or tuple.
    # If tuple, use -1 to show 'no upper bound'.
    DIM = (1, -1)

    def __init__(self, n: int = DIM[0]) -> None:
        super().__init__(n)

    @staticmethod
    def evaluate(x, xp=None):
        """
        Evaluate the Schwefel function.

        :param x: n-d input point (array-like)
        :param xp: Optional array API namespace (e.g., numpy, Torch)
        :return: Scalar function output
        """
        if xp is None:
            if isinstance(x, list):
                xp = np
            else:
                xp = array_api_compat.array_namespace(x)

        abs_fn = _get_abs(xp)

        d = len(x)
        term1 = 418.9829 * d
        if hasattr(xp, "asarray"):
            x = xp.asarray(x)
            term2 = xp.sum(x * xp.sin(xp.sqrt(abs_fn(x))))
        else:
            term2 = sum(x[i] * xp.sin(xp.sqrt(abs_fn(x[i]))) for i in range(d))

        res = term1 - term2

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
        return [[-500.0, 500.0] for i in range(self._ndims)]

    def argmin(self):
        """
        Returns function argmin.

        :return: List of minimizer(s)
        """
        return [[420.968746 for i in range(self._ndims)]]

@tag(["VLSE", "Many_local_minima", "Unconstrained", "Multimodal", "2D", "Continuous", "Differentiable"])
class Shubert(BenchmarkFunction):
    """
    The Shubert function is a 2D function with many local minima and
    18 Global minima.

    :References: https://www.sfu.ca/~ssurjano/shubert.html
    """

    # Acceptable dimensions. Either integer or tuple.
    # If tuple, use -1 to show 'no upper bound'.
    DIM = 2

    def __init__(self, n: int = 2) -> None:
        super().__init__(n)

    @staticmethod
    def evaluate(x,  xp=None):
        """
        Evaluate the Shubert function.

        :param x: 2D input point (array-like)
        :param xp: Optional array API namespace (e.g., numpy, Torch)
        :return: Scalar function output
        """
        # Unpack the input vector
        x1 = x[0]
        x2 = x[1]

        if xp is None:
            if isinstance(x, list):
                xp = np
            else:
                xp = array_api_compat.array_namespace(x)

        # Compute the Shubert function
        term1 = sum([i * xp.cos((i + 1) * x1 + i) for i in range(1, 6)])
        term2 = sum([i * xp.cos((i + 1) * x2 + i) for i in range(1, 6)])

        res = term1 * term2

        return res

    @staticmethod
    def min():
        """
        Returns known minimum function value.

        :return: Minimum value (float)
        """
        return -186.7309

    @staticmethod
    def bounds():
        """
        Returns problem bounds.

        :return: List of [lower, upper] for each dimension
        """
        return [[-10, 10], [-10, 10]]

    @staticmethod
    def argmin():
        """
        Returns function argmin.

        :return: List of minimizer(s)
        """
        return [[-7.0835, 4.8580], [-7.0835, -7.7083],
                [-1.4251, -7.0835], [5.4828, 4.8580],
                [-1.4251, -0.8003], [4.8580, 5.4828],
                [-7.7083, -7.0835], [-7.0835, -1.4251],
                [-7.7083, -0.8003], [-7.7083, 5.4828],
                [-0.8003, -7.7083], [-0.8003, -1.4251],
                [-0.8003, 4.8580], [-1.4251, 5.4828],
                [5.4828, -7.7083], [4.8580, -7.0835],
                [5.4828, -1.4251], [4.8580, -0.8003]]

@tag(["VLSE", "Bowl_shaped", "Unconstrained", "Multimodal", "2D", "Continuous", "Differentiable", "Separable"])
class Bohachevsky1(BenchmarkFunction):
    """
    The Bohachevsky functions are bowl-shaped.

    :References: https://www.sfu.ca/~ssurjano/boha.html
    """

    # Acceptable dimensions. Either integer or tuple.
    # If tuple, use -1 to show 'no upper bound'.
    DIM = 2

    def __init__(self, n: int = 2) -> None:
        super().__init__(n)

    @staticmethod
    def evaluate(x, xp=None):
        """
        Evaluate the Bohachevsky 1 function.

        :param x: 2D input point (array-like)
        :param xp: Optional array API namespace (e.g., numpy, Torch)
        :return: Scalar function output
        """
        # Unpack the input vector
        x1 = x[0]
        x2 = x[1]

        if xp is None:
            if isinstance(x, list):
                xp = np
            else:
                xp = array_api_compat.array_namespace(x)

        # Compute the Bohachevsky function
        term1 = x1**2 + 2 * x2**2
        term2 = 0.3 * xp.cos(3 * np.pi * x1) + 0.4 * xp.cos(4 * np.pi * x2)

        res = term1 - term2 + 0.7

        return res

    @staticmethod
    def min():
        """
        Returns known minimum function value.

        :return: Minimum value (float)
        """
        return 0.0

    @staticmethod
    def bounds():
        """
        Returns problem bounds.

        :return: List of [lower, upper] for each dimension
        """
        return [[-100.0, 100.0], [-100.0, 100.0]]

    @staticmethod
    def argmin():
        """
        Returns function argmin.

        :return: List of minimizer(s)
        """
        return [[0.0, 0.0]]

@tag(["VLSE", "Bowl_shaped", "Unconstrained", "Multimodal", "2D", "Continuous", "Differentiable", "Non_separable"])
class Bohachevsky2(BenchmarkFunction):
    """
    The Bohachevsky functions are bowl-shaped.

    :References: https://www.sfu.ca/~ssurjano/boha.html
    """

    # Acceptable dimensions. Either integer or tuple.
    # If tuple, use -1 to show 'no upper bound'.
    DIM = 2

    def __init__(self, n: int = 2) -> None:
        super().__init__(n)

    @staticmethod
    def evaluate(x, xp=None):
        """
        Evaluate the Bohachevsky 2 function.

        :param x: 2D input point (array-like)
        :param xp: Optional array API namespace (e.g., numpy, Torch)
        :return: Scalar function output
        """
        # Unpack the input vector
        x1 = x[0]
        x2 = x[1]

        if xp is None:
            if isinstance(x, list):
                xp = np
            else:
                xp = array_api_compat.array_namespace(x)

        # Compute the Bohachevsky function
        term1 = x1**2 + 2 * x2**2
        term2 = 0.3 * xp.cos(3 * np.pi * x1) * xp.cos(4 * np.pi * x2)

        res = term1 - term2 + 0.3

        return res

    @staticmethod
    def min():
        """
        Returns known minimum function value.

        :return: Minimum value (float)
        """
        return 0.0

    @staticmethod
    def bounds():
        """
        Returns problem bounds.

        :return: List of [lower, upper] for each dimension
        """
        return [[-100.0, 100.0], [-100.0, 100.0]]

    @staticmethod
    def argmin():
        """
        Returns function argmin.

        :return: List of minimizer(s)
        """
        return [[0.0, 0.0]]

@tag(["VLSE", "Bowl_shaped", "Unconstrained", "Multimodal", "2D", "Continuous", "Differentiable", "Non_separable"])
class Bohachevsky3(BenchmarkFunction):
    """
    The Bohachevsky functions are bowl-shaped.

    :References: https://www.sfu.ca/~ssurjano/boha.html
    """

    # Acceptable dimensions. Either integer or tuple.
    # If tuple, use -1 to show 'no upper bound'.
    DIM = 2

    def __init__(self, n: int = 2) -> None:
        super().__init__(n)

    @staticmethod
    def evaluate(x, xp=None):
        """
        Evaluate the Bohachevsky 3 function.

        :param x: 2D input point (array-like)
        :param xp: Optional array API namespace (e.g., numpy, Torch)
        :return: Scalar function output
        """
        # Unpack the input vector
        x1 = x[0]
        x2 = x[1]

        if xp is None:
            if isinstance(x, list):
                xp = np
            else:
                xp = array_api_compat.array_namespace(x)

        # Compute the Bohachevsky function
        term1 = x1**2 + 2 * x2**2
        term2 = 0.3 * xp.cos(3 * np.pi * x1 + 4 * np.pi * x2)

        res = term1 - term2 + 0.3

        return res

    @staticmethod
    def min():
        """
        Returns known minimum function value.

        :return: Minimum value (float)
        """
        return 0.0

    @staticmethod
    def bounds():
        """
        Returns problem bounds.

        :return: List of [lower, upper] for each dimension
        """
        return [[-100.0, 100.0], [-100.0, 100.0]]

    @staticmethod
    def argmin():
        """
        Returns function argmin.

        :return: List of minimizer(s)
        """
        return [[0.0, 0.0]]

@tag(["VLSE", "Bowl_shaped", "Unconstrained", "Multimodal", "nD", "Continuous", "Differentiable", "Non_separable"])
class Perm0(BenchmarkFunction):
    """
    The perm0 function is bowl-shaped.

    :References: https://www.sfu.ca/~ssurjano/perm0db.html
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

        d = len(x)
        beta = 10

        res = 0.0
        for i in range(d):
            inner = 0.0
            for j in range(d):
                inner += (j + 1 + beta) * ((x[j] ** (i + 1)) - (1 / (j + 1)) ** (i + 1))
            res += inner ** 2
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
        return [[-self._ndims, self._ndims] for i in range(self._ndims)]

    def argmin(self):
        """
        Returns function argmin.

        :return: List of minimizer(s)
        """
        return [[1 / (i + 1) for i in range(self._ndims)]]

@tag(["VLSE", "Bowl_shaped", "Unconstrained", "Unimodal", "nD", "Continuous", "Differentiable", "Separable"])
class Rothyp(BenchmarkFunction):
    """
    The Rotated Hyper-Ellipsoid function is a simple continuous,
    convex, and unimodal nD function.

    :References: https://www.sfu.ca/~ssurjano/rothyp.html
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

        d = len(x)
        res = 0.0
        for i in range(d):
            for j in range(i):
                res += x[j]**2

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
        return [[-65.536, 65.536] for i in range(self._ndims)]

    def argmin(self):
        """
        Returns function argmin.

        :return: List of minimizer(s)
        """
        return [[0.0 for _ in range(self._ndims)]]

@tag(["VLSE", "Bowl_shaped", "Unconstrained", "Unimodal", "nD", "Continuous", "Differentiable", "Separable"])
class Sphere(BenchmarkFunction):
    """
    The Sphere function is a simple continuous,
    convex, and unimodal nD function.

    :References: https://www.sfu.ca/~ssurjano/spheref.html
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

        d = len(x)
        res = 0.0

        for i in range(d):
            res += x[i] ** 2

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
        return [[-5.12, 5.12] for i in range(self._ndims)]

    def argmin(self):
        """
        Returns function argmin.

        :return: List of minimizer(s)
        """
        return [[0.0 for i in range(self._ndims)]]

@tag(["VLSE", "Bowl_shaped", "Unconstrained", "Unimodal", "nD", "Separable"])
class SumPow(BenchmarkFunction):
    """
    The sum of different powers function is a unimodal function.

    :Reference: https://www.sfu.ca/~ssurjano/sumpow.html
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

        abs_fn = _get_abs(xp)
        d = len(x)
        res = 0.0
        for i in range(d):
            res += abs_fn(x[i]) ** (i + 2)
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
        return [[-1, 1] for i in range(self._ndims)]

    def argmin(self):
        """
        Returns function argmin.

        :return: List of minimizer(s)
        """
        return [[0.0 for i in range(self._ndims)]]

@tag(["VLSE", "Bowl_shaped", "Unconstrained", "Unimodal", "nD", "Continuous", "Differentiable", "Separable"])
class SumSq(BenchmarkFunction):
    """
    The sum of squares function is a unimodal function.

    :Reference: https://www.sfu.ca/~ssurjano/sumsqu.html
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

        d = len(x)
        res = 0.0
        for i in range(d):
            res += (i + 1) * x[i] ** 2
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
        return [[-10, 10] for i in range(self._ndims)]

    def argmin(self):
        """
        Returns function argmin.

        :return: List of minimizer(s)
        """
        return [[0.0 for i in range(self._ndims)]]

@tag(["VLSE", "Bowl_shaped", "Unconstrained", "Unimodal", "nD", "Continuous", "Differentiable", "Non_separable"])
class Trid(BenchmarkFunction):
    """
    The Trid function is a unimodal function.

    :Reference: https://www.sfu.ca/~ssurjano/trid.html
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

        d = len(x)
        term1 = (x[0] - 1) ** 2
        term2 = 0.0

        for i in range(1, d):
            term1 += (x[i] - 1) ** 2
            term2 += x[i] * x[i - 1]
        return term1 - term2

    def min(self):
        """
        Returns known minimum function value.

        :return: Minimum value (float)
        """
        d = self._ndims
        return -d * (d + 4) * (d - 1) / 6

    def bounds(self):
        """
        Returns problem bounds.

        :return: List of [lower, upper] for each dimension
        """
        d = self._ndims
        return [[-d**2, d**2] for i in range(self._ndims)]

    def argmin(self):
        """
        Returns function argmin.

        :return: List of minimizer(s)
        """
        d = self._ndims
        return [[(i + 1)*(d + 1 - (i + 1)) for i in range(self._ndims)]]

@tag(["VLSE", "Plate_shaped", "Unconstrained", "Unimodal", "2D", "Continuous", "Differentiable", "Non_separable"])
class Booth(BenchmarkFunction):
    """
    The Booth function is a 2 dimensional unimodal function.

    :Reference: https://www.sfu.ca/~ssurjano/booth.html
    """

    # Acceptable dimensions. Either integer or tuple.
    # If tuple, use -1 to show 'no upper bound'.
    DIM = 2

    def __init__(self, n: int = 2) -> None:
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

        return (x[0] + 2 * x[1] - 7) ** 2 + (2 * x[0] + x[1] - 5) ** 2

    @staticmethod
    def min():
        """
        Returns known minimum function value.

        :return: Minimum value (float)
        """
        return 0.0

    @staticmethod
    def bounds():
        """
        Returns problem bounds.

        :return: List of [lower, upper] for each dimension
        """
        return [[-10, 10], [-10, 10]]

    @staticmethod
    def argmin():
        """
        Returns function argmin.

        :return: List of minimizer(s)
        """
        return [[1, 3]]

@tag(["VLSE", "Plate_shaped", "Unconstrained", "Unimodal", "2D", "Continuous", "Differentiable", "Non_separable"])
class Matyas(BenchmarkFunction):
    """
    The Matyas function is a 2 dimensional unimodal function.

    :Reference: https://www.sfu.ca/~ssurjano/matya.html
    """

    # Acceptable dimensions. Either integer or tuple.
    # If tuple, use -1 to show 'no upper bound'.
    DIM = 2

    def __init__(self, n: int = 2) -> None:
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

        return 0.26 * (x[0]**2 + x[1]**2) - 0.48 * x[0] * x[1]

    @staticmethod
    def min():
        """
        Returns known minimum function value.

        :return: Minimum value (float)
        """
        return 0.0

    @staticmethod
    def bounds():
        """
        Returns problem bounds.

        :return: List of [lower, upper] for each dimension
        """
        return [[-10, 10], [-10, 10]]

    @staticmethod
    def argmin():
        """
        Returns function argmin.

        :return: List of minimizer(s)
        """
        return [[0, 0]]

@tag(["VLSE", "Plate_shaped", "Unconstrained", "Multimodal", "2D", "Continuous", "Differentiable", "Non_separable"])
class McCormick(BenchmarkFunction):
    """
    The McCormick function is a 2 dimensional multimodal function.

    :Reference: https://www.sfu.ca/~ssurjano/mccorm.html
                https://infinity77.net/global_optimization/test_functions_nd_M.html#go_benchmark.McCormick
    """

    # Acceptable dimensions. Either integer or tuple.
    # If tuple, use -1 to show 'no upper bound'.
    DIM = 2

    def __init__(self, n: int = 2) -> None:
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

        return xp.sin(x[0] + x[1]) + (x[0] - x[1])**2 - 1.5 * x[0] + 2.5 * x[1] + 1

    @staticmethod
    def min():
        """
        Returns known minimum function value.

        :return: Minimum value (float)
        """
        return -1.913222954981037

    @staticmethod
    def bounds():
        """
        Returns problem bounds.

        :return: List of [lower, upper] for each dimension
        """
        return [[-1.5, 4], [-3, 4]]

    @staticmethod
    def argmin():
        """
        Returns function argmin.

        :return: List of minimizer(s)
        """
        return [[-0.5471975602214493, -1.547197559268372]]

@tag(["VLSE", "Plate_shaped", "Unconstrained", "Multimodal", "4D", "Continuous", "Differentiable", "Non_separable"])
class PowerSum(BenchmarkFunction):
    """
    The Power Sum function is a 4 dimensional multimodal function.

    :Reference: https://www.sfu.ca/~ssurjano/powersum.html
                http://infinity77.net/global_optimization/test_functions_nd_P.html#go_benchmark.PowerSum
    """

    # Acceptable dimensions. Either integer or tuple.
    # If tuple, use -1 to show 'no upper bound'.
    DIM = 4

    def __init__(self, n: int = 4) -> None:
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

        b = [8, 18, 44, 114]
        res = 0.0
        for i in range(4):
            inner = 0.0
            for j in range(4):
                inner += x[j] ** (i + 1)
            res += (inner - b[i])**2
        return res

    @staticmethod
    def min():
        """
        Returns known minimum function value.

        :return: Minimum value (float)
        """
        return 0.0

    @staticmethod
    def bounds():
        """
        Returns problem bounds.

        :return: List of [lower, upper] for each dimension
        """
        return [[0, 4] for _ in range(4)]

    @staticmethod
    def argmin():
        """
        Returns function argmin.

        :return: List of minimizer(s)
        """
        return [[1, 2, 2, 3]]

@tag(["VLSE", "Plate_shaped", "Unconstrained", "Unimodal", "nD", "Continuous", "Differentiable", "Non_separable"])
class Zakharov(BenchmarkFunction):
    """
    The Zakharov function is a n dimensional unimodal function.

    :Reference: https://www.sfu.ca/~ssurjano/zakharov.html
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

        term1 = 0.0
        inner = 0.0

        d = len(x)
        for i in range(d):
            term1 += x[i] ** 2
            inner += 0.5 * (i + 1) * x[i]

        return term1 + inner ** 2 + inner ** 4

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
        return [[0.0 for _ in range(self._ndims)]]

@tag(["VLSE", "Valley_shaped", "Unconstrained", "Multimodal", "2D", "Continuous", "Differentiable", "Non_separable"])
class Camel3(BenchmarkFunction):
    """
    The Three-Hump Camel function is a 2 dimensional function with three local
    minima and one global minimum.

    :Reference: https://www.sfu.ca/~ssurjano/camel3.html
    """

    # Acceptable dimensions. Either integer or tuple.
    # If tuple, use -1 to show 'no upper bound'.
    DIM = 2

    def __init__(self, n: int = 2) -> None:
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

        x1 = x[0]
        x2 = x[1]

        return 2 * x1**2 - 1.05 * x1**4 + (x1**6) / 6 + x1 * x2 + x2**2

    @staticmethod
    def min():
        """
        Returns known minimum function value.

        :return: Minimum value (float)
        """
        return 0.0

    @staticmethod
    def bounds():
        """
        Returns problem bounds.

        :return: List of [lower, upper] for each dimension
        """
        return [[-5, 5], [-5, 5]]

    @staticmethod
    def argmin():
        """
        Returns function argmin.

        :return: List of minimizer(s)
        """
        return [[0.0, 0.0]]

@tag(["VLSE", "Valley_shaped", "Unconstrained", "Multimodal", "2D", "Continuous", "Differentiable", "Non_separable"])
class Camel6(BenchmarkFunction):
    """
    The Six-Hump Camel function is a 2 dimensional function with six local
    minima and two global minimum.

    :Reference: https://www.sfu.ca/~ssurjano/camel6.html
                http://infinity77.net/global_optimization/test_functions_nd_S.html#go_benchmark.SixHumpCamel
    """

    # Acceptable dimensions. Either integer or tuple.
    # If tuple, use -1 to show 'no upper bound'.
    DIM = 2

    def __init__(self, n: int = 2) -> None:
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

        x1 = x[0]
        x2 = x[1]

        term1 = (4 - 2.1 * x1**2 + x1**4 / 3) * x1**2
        term2 = x1 * x2
        term3 = (-4 + 4 * x2**2) * x2**2

        return term1 + term2 + term3

    @staticmethod
    def min():
        """
        Returns known minimum function value.

        :return: Minimum value (float)
        """
        return -1.031628453489877

    @staticmethod
    def bounds():
        """
        Returns problem bounds.

        :return: List of [lower, upper] for each dimension
        """
        return [[-3, 3], [-2, 2]]

    @staticmethod
    def argmin():
        """
        Returns function argmin.

        :return: List of minimizer(s)
        """
        return [[0.08984201368301331, -0.7126564032704135], [-0.08984201368301331, 0.7126564032704135]]

@tag(["VLSE", "Valley_shaped", "Unconstrained", "Multimodal", "nD", "Continuous", "Differentiable", "Non_separable"])
class DixonPrice(BenchmarkFunction):
    """
    The Dixon-Price function is a n dimensional unimodal function.

    :Reference: https://www.sfu.ca/~ssurjano/dixonpr.html
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

        term1 = (x[0] - 1) ** 2
        term2 = 0.0

        d = len(x)
        for i in range(1, d):
            term2 += (i + 1) * (2 * x[i] ** 2 - x[i - 1]) ** 2

        return term1 + term2

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
        return [[-10, 10] for _ in range(self._ndims)]

    def argmin(self):
        """
        Returns function argmin.

        :return: List of minimizer(s)
        """
        return [[(2**(-(2**(i + 1) - 2) / 2**(i+1))) for i in range(self._ndims)]]

@tag(["VLSE", "Valley_shaped", "Unconstrained", "Multimodal", "nD", "Continuous", "Differentiable", "Non_separable"])
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

@tag(["VLSE", "Steep_ridges_drops", "Unconstrained", "Multimodal", "2D", "Discontinuous", "Non_separable"])
class Dejong5(BenchmarkFunction):
    """
    The De Jong 5 function is a multimodal function with many local minima
    and very sharp drops making it semi-discontinuous.
    The global minimum is unknown.

    :References: https://www.sfu.ca/~ssurjano/dejong5.html
    """

    # Acceptable dimensions. Either integer or tuple.
    # If tuple, use -1 to show 'no upper bound'.
    DIM = 2

    def __init__(self, n: int = 2) -> None:
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

        x1 = x[0]
        x2 = x[1]

        expr = 0.002
        for i in range(-2, 3):
            for j in range(-2, 3):
                denom = (
                    5 * (i + 2) + j + 3 +
                    (x1 - 16 * j) ** 6 +
                    (x2 - 16 * i) ** 6
                )
                expr += 1 / denom
        return 1 / expr

    @staticmethod
    def min():
        """
        Minimum function value unknown.

        :return: None
        """
        return None

    @staticmethod
    def bounds():
        """
        Returns problem bounds.

        :return: List of [lower, upper] for each dimension
        """
        return [[-65.536, 65.536], [-65.536, 65.536]]

    @staticmethod
    def argmin():
        """
        Function argmin is unknown.

        :return: None
        """
        return None

@tag(["VLSE", "Steep_ridges_drops", "Unconstrained", "Multimodal", "2D", "Discontinuous", "Differentiable", "Non_separable"])
class Easom(BenchmarkFunction):
    """
    The Easom function is a 2 dimensional function with mulitple local minima
    and one global minimum.

    :Reference: https://www.sfu.ca/~ssurjano/easom.html
    """

    # Acceptable dimensions. Either integer or tuple.
    # If tuple, use -1 to show 'no upper bound'.
    DIM = 2

    def __init__(self, n: int = 2) -> None:
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

        x1 = x[0]
        x2 = x[1]

        return -xp.cos(x1) * xp.cos(x2) * xp.exp(-(x1 - np.pi)**2 - (x2 - np.pi)**2)

    @staticmethod
    def min():
        """
        Returns known minimum function value.

        :return: Minimum value (float)
        """
        return -1

    @staticmethod
    def bounds():
        """
        Returns problem bounds.

        :return: List of [lower, upper] for each dimension
        """
        return [[-100, 100], [-100, 100]]

    @staticmethod
    def argmin():
        """
        Returns function argmin.

        :return: List of minimizer(s)
        """
        return [[np.pi, np.pi]]

@tag(["VLSE", "Steep_ridges_drops", "Unconstrained", "Multimodal", "nD", "Continuous", "Differentiable", "Non_separable"])
class Michalewicz(BenchmarkFunction):
    """
    The Michalewicz function is a n dimensional multimodal function.

    :Reference: https://www.sfu.ca/~ssurjano/michal.html
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
        m = 10
        d = len(x)

        for i in range(d):
            res += xp.sin(x[i]) * xp.sin((i + 1) * x[i]**2 / np.pi)**(2 * m)

        return -res

    def min(self):
        """
        Returns known minimum function value.

        :return: Minimum value (float)
        """
        if self._ndims == 2:
            return -1.8013
        elif self._ndims == 5:
            return -4.687658
        elif self._ndims == 10:
            return -9.66015
        else:
            return None

    def bounds(self):
        """
        Returns problem bounds.

        :return: List of [lower, upper] for each dimension
        """
        return [[0, np.pi] for _ in range(self._ndims)]

    def argmin(self):
        """
        Returns function argmin.

        :return: List of minimizer(s)
        """
        if self._ndims == 2:
            return [[2.202906, 1.570796]]
        else:
            return None

@tag(["VLSE", "Other", "Unconstrained", "Multimodal", "2D", "Continuous", "Differentiable", "Non_separable"])
class Beale(BenchmarkFunction):
    """
    The Beale function 2D multimodal function with peaks at the corners of the
    domain.

    :References: https://www.sfu.ca/~ssurjano/beale.html
    """

    # Acceptable dimensions. Either integer or tuple.
    # If tuple, use -1 to show 'no upper bound'.
    DIM = 2

    def __init__(self, n: int = 2) -> None:
        super().__init__(n)

    @staticmethod
    def evaluate(x, xp=None):
        """
        Evaluate the Beale function.

        :param x: 2D input point (array-like)
        :param xp: Optional array API namespace (e.g., numpy, Torch)
        :return: Scalar function output
        """
        if xp is None:
            if isinstance(x, list):
                xp = np
            else:
                xp = array_api_compat.array_namespace(x)

        x1 = x[0]
        x2 = x[1]
        term1 = (1.5 - x1 + x1 * x2) ** 2
        term2 = (2.25 - x1 + x1 * x2**2) ** 2
        term3 = (2.625 - x1 + x1 * x2**3) ** 2
        return term1 + term2 + term3

    @staticmethod
    def min():
        """
        Returns known minimum function value.

        :return: Minimum value (float)
        """
        return 0.0

    @staticmethod
    def bounds():
        """
        Returns problem bounds.

        :return: List of [lower, upper] for each dimension
        """
        return [[-4.5, 4.5], [-4.5, 4.5]]

    @staticmethod
    def argmin():
        """
        Returns function argmin.

        :return: List of minimizer(s)
        """
        return [[3, 0.5]]

@tag(["VLSE", "Other", "Unconstrained", "Multimodal", "2D", "Continuous", "Differentiable", "Non_separable"])
class Branin(BenchmarkFunction):
    """
    The Branin function 2D function with three global minima.

    :References: https://www.sfu.ca/~ssurjano/branin.html
                 http://infinity77.net/global_optimization/test_functions_nd_B.html#go_benchmark.Branin01
    """

    # Acceptable dimensions. Either integer or tuple.
    # If tuple, use -1 to show 'no upper bound'.
    DIM = 2

    def __init__(self, n: int = 2) -> None:
        super().__init__(n)

    @staticmethod
    def evaluate(x, xp=None):
        """
        Evaluate the Beale function.

        :param x: 2D input point (array-like)
        :param xp: Optional array API namespace (e.g., numpy, Torch)
        :return: Scalar function output
        """
        if xp is None:
            if isinstance(x, list):
                xp = np
            else:
                xp = array_api_compat.array_namespace(x)

        x1 = x[0]
        x2 = x[1]

        a = 1.0
        b = 5.1 / (4 * np.pi**2)
        c = 5 / np.pi
        r = 6
        s = 10
        t = 1 / (8 * np.pi)

        term1 = a * (x2 - b * x1**2 + c * x1 - r)**2
        term2 = s * (1 - t) * xp.cos(x1)
        return term1 + term2 + s

    @staticmethod
    def min():
        """
        Returns known minimum function value.

        :return: Minimum value (float)
        """
        return 0.39788735772973816

    @staticmethod
    def bounds():
        """
        Returns problem bounds.

        :return: List of [lower, upper] for each dimension
        """
        return [[-5, 10], [0, 15]]

    @staticmethod
    def argmin():
        """
        Returns function argmin.

        :return: List of minimizer(s)
        """
        return [[-np.pi, 12.275], [np.pi, 2.275], [9.42478, 2.475]]

@tag(["VLSE", "Other", "Unconstrained", "Multimodal", "4D", "Continuous", "Differentiable", "Non_separable"])
class Colville(BenchmarkFunction):
    """
    The Colville function is a 4 dimensional multimodal function.

    :Reference: https://www.sfu.ca/~ssurjano/colville.html
    """

    # Acceptable dimensions. Either integer or tuple.
    # If tuple, use -1 to show 'no upper bound'.
    DIM = 4

    def __init__(self, n: int = 4) -> None:
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

        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        x4 = x[3]

        term1 = 100 * (x1**2 - x2)**2
        term2 = (x1 - 1)**2
        term3 = (x3 - 1)**2
        term4 = 90 * (x3**2 - x4)**2
        term5 = 10.1 * ((x2 - 1)**2 + (x4 - 1)**2)
        term6 = 19.8 * (x2 - 1) * (x4 - 1)
        return term1 + term2 + term3 + term4 + term5 + term6

    @staticmethod
    def min():
        """
        Returns known minimum function value.

        :return: Minimum value (float)
        """
        return 0.0

    @staticmethod
    def bounds():
        """
        Returns problem bounds.

        :return: List of [lower, upper] for each dimension
        """
        return [[-10, 10] for _ in range(4)]

    @staticmethod
    def argmin():
        """
        Returns function argmin.

        :return: List of minimizer(s)
        """
        return [[1, 1, 1, 1]]

@tag(["VLSE", "Other", "Unconstrained", "Multimodal", "1D", "Continuous", "Differentiable"])
class Forrester(BenchmarkFunction):
    """
    The Forrester Et Al. function is a multimodal 1D function with
    one global minimum.

    :References: https://www.sfu.ca/~ssurjano/grlee12.html
    """

    # Acceptable dimensions. Either integer or tuple.
    # If tuple, use -1 to show 'no upper bound'.
    DIM = 1

    def __init__(self, n: int = 1) -> None:
        super().__init__(n)

    @staticmethod
    def evaluate(x, xp=None):
        """
        Evaluate the Gramacy-Lee function.

        :param x: 1D input point
        :param xp: Optional array API namespace (e.g., numpy, Torch)
        :return: Scalar function output
        """
        if xp is not None:
            x = x[0]
        if xp is None:
            if isinstance(x, list):
                xp = np
            else:
                xp = array_api_compat.array_namespace(x)

        return (6 * x -2)**2 * xp.sin(12 * x - 4)

    @staticmethod
    def min():
        """
        Returns known minimum function value.

        :return: Minimum value (float)
        """
        return -6.02074

    @staticmethod
    def bounds():
        """
        Returns problem bounds.

        :return: List of [lower, upper] for each dimension
        """
        return [[0, 1]]

    @staticmethod
    def argmin():
        """
        Returns function argmin.

        :return: List of minimizer(s)
        """
        return [[0.75725]]

@tag(["VLSE", "Other", "Unconstrained", "Multimodal", "2D", "Continuous", "Differentiable", "Non_separable"])
class GoldsteinPrice(BenchmarkFunction):
    """
    The Goldstein-Price function is a 2D multimodal function.

    :References: https://www.sfu.ca/~ssurjano/goldpr.html
    """

    # Acceptable dimensions. Either integer or tuple.
    # If tuple, use -1 to show 'no upper bound'.
    DIM = 2

    def __init__(self, n: int = 2) -> None:
        super().__init__(n)

    @staticmethod
    def evaluate(x, xp=None):
        """
        Evaluate the Goldstein-Price function.

        :param x: 2D input point (array-like)
        :param xp: Optional array API namespace (e.g., numpy, Torch)
        :return: Scalar function output
        """
        if xp is None:
            if isinstance(x, list):
                xp = np
            else:
                xp = array_api_compat.array_namespace(x)

        x1 = x[0]
        x2 = x[1]

        term1 = 1 + (x1 + x2 + 1)**2 * (19 - 14 * x1 + 3 * x1**2 - 14 * x2 + 6 * x1 * x2 + 3 * x2**2)
        term2 = 30 + (2 * x1 - 3 * x2)**2 * (18 - 32 * x1 + 12 * x1**2 + 48 * x2 - 36 * x1 * x2 + 27 * x2**2)
        return term1 * term2

    @staticmethod
    def min():
        """
        Returns known minimum function value.

        :return: Minimum value (float)
        """
        return 3.0

    @staticmethod
    def bounds():
        """
        Returns problem bounds.

        :return: List of [lower, upper] for each dimension
        """
        return [[-2, 2], [-2, 2]]

    @staticmethod
    def argmin():
        """
        Returns function argmin.

        :return: List of minimizer(s)
        """
        return [[0, -1]]

@tag(["VLSE", "Other", "Unconstrained", "Multimodal", "3D", "Continuous", "Differentiable", "Non_separable"])
class Hartmann3D(BenchmarkFunction):
    """
    The 3D Hartmann function has four local minima and one global minimum.

    :Reference: https://www.sfu.ca/~ssurjano/hart3.html
                http://infinity77.net/global_optimization/test_functions_nd_H.html#go_benchmark.Hartmann3
    """

    # Acceptable dimensions. Either integer or tuple.
    # If tuple, use -1 to show 'no upper bound'.
    DIM = 3

    def __init__(self, n: int = 3) -> None:
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

        alpha = [1.0, 1.2, 3.0, 3.2]
        A = [[3.0, 10.0, 30.0],
             [0.1, 10.0, 35.0],
             [3.0, 10.0, 30.0],
             [0.1, 10.0, 35.0]]
        P = [[3689.0, 1170.0, 2673.0],
             [4699.0, 4387.0, 7470.0],
             [1091.0, 8732.0, 5547.0],
             [381.0, 5743.0, 8828.0]]

        res = 0.0
        for i in range(4):
            inner = 0.0
            for j in range(3):
                inner += A[i][j] * (x[j] - (1e-4)*P[i][j])**2
            res += alpha[i] * xp.exp(-inner)

        return -res

    @staticmethod
    def min():
        """
        Returns known minimum function value.

        :return: Minimum value (float)
        """
        return -3.86278

    @staticmethod
    def bounds():
        """
        Returns problem bounds.

        :return: List of [lower, upper] for each dimension
        """
        return [[0, 1] for _ in range(3)]

    @staticmethod
    def argmin():
        """
        Returns function argmin.

        :return: List of minimizer(s)
        """
        return [[0.114614, 0.555649, 0.852547]]

@tag(["VLSE", "Other", "Unconstrained", "Multimodal", "4D", "Continuous", "Differentiable", "Non_separable"])
class Hartmann4D(BenchmarkFunction):
    """
    The 4D Hartmann function is multimodal.

    :Reference: https://www.sfu.ca/~ssurjano/Code/hart4m.html
    """

    # Acceptable dimensions. Either integer or tuple.
    # If tuple, use -1 to show 'no upper bound'.
    DIM = 4

    def __init__(self, n: int = 4) -> None:
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

        alpha = [1.0, 1.2, 3.0, 3.2]
        A = [[10.0, 3.0, 17.0, 3.5, 1.7, 8.0],
            [0.05, 10.0, 17.0, 0.1, 8.0, 14.0],
            [3.0, 3.5, 1.7, 10.0, 17.0, 8.0],
            [17.0, 8.0, 0.05, 10.0, 0.1, 14.0]]
        P = [[1312.0, 1696.0, 5569.0, 124.0, 8283.0, 5886.0],
            [2329.0, 4135.0, 8307.0, 3736.0, 1004.0, 9991.0],
            [2348.0, 1451.0, 3522.0, 2883.0, 3047.0, 6650.0],
            [4047.0, 8828.0, 8732.0, 5743.0, 1091.0, 381.0]]

        res = 0.0
        for i in range(4):
            inner = 0.0
            for j in range(4):
                inner += A[i][j] * (x[j] - (1e-4)*P[i][j])**2
            res += alpha[i] * xp.exp(-inner)

        return (1.1 - res) / 0.839

    @staticmethod
    def min():
        """
        Returns known minimum function value.

        :return: Minimum value (float)
        """
        return None

    @staticmethod
    def bounds():
        """
        Returns problem bounds.

        :return: List of [lower, upper] for each dimension
        """
        return [[0, 1] for _ in range(4)]

    @staticmethod
    def argmin():
        """
        Returns function argmin.

        :return: List of minimizer(s)
        """
        return None

@tag(["VLSE", "Other", "Unconstrained", "Multimodal", "6D", "Continuous", "Differentiable", "Non_separable"])
class Hartmann6D(BenchmarkFunction):
    """
    The 6D Hartmann function is multimodal.

    :Reference: https://www.sfu.ca/~ssurjano/hart6.html
    """

    # Acceptable dimensions. Either integer or tuple.
    # If tuple, use -1 to show 'no upper bound'.
    DIM = 6

    def __init__(self, n: int = 6) -> None:
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

        alpha = [1.0, 1.2, 3.0, 3.2]
        A = [[10.0, 3.0, 17.0, 3.5, 1.7, 8.0],
            [0.05, 10.0, 17.0, 0.1, 8.0, 14.0],
            [3.0, 3.5, 1.7, 10.0, 17.0, 8.0],
            [17.0, 8.0, 0.05, 10.0, 0.1, 14.0]]
        P = [[1312.0, 1696.0, 5569.0, 124.0, 8283.0, 5886.0],
            [2329.0, 4135.0, 8307.0, 3736.0, 1004.0, 9991.0],
            [2348.0, 1451.0, 3522.0, 2883.0, 3047.0, 6650.0],
            [4047.0, 8828.0, 8732.0, 5743.0, 1091.0, 381.0]]

        res = 0.0
        for i in range(4):
            inner = 0.0
            for j in range(6):
                inner += A[i][j] * (x[j] - (1e-4)*P[i][j])**2
            res += alpha[i] * xp.exp(-inner)

        return -res

    @staticmethod
    def min():
        """
        Returns known minimum function value.

        :return: Minimum value (float)
        """
        return -3.32237

    @staticmethod
    def bounds():
        """
        Returns problem bounds.

        :return: List of [lower, upper] for each dimension
        """
        return [[0, 1] for _ in range(6)]

    @staticmethod
    def argmin():
        """
        Returns function argmin.

        :return: List of minimizer(s)
        """
        return [[0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573]]

@tag(["VLSE", "Other", "Unconstrained", "Multimodal", "nD", "Continuous", "Differentiable", "Non_separable"])
class Perm(BenchmarkFunction):
    """
    The n Dimensional perm function is multimodal. This implementation
    uses beta = 0.5.

    :Reference: https://www.sfu.ca/~ssurjano/michal.html
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
        b = 0.5
        d = len(x)

        for i in range(d):
            inner = 0.0
            for j in range(d):
                inner += ( ((j + 1) ** (i + 1) + b) * ((x[j] / (j + 1)) - 1))**2
            res += inner

        return res

    def min(self):
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
        d = self._ndims
        return [[-d, d] for _ in range(d)]

    def argmin(self):
        """
        Returns function argmin.

        :return: List of minimizer(s)
        """
        res = []
        for i in range(self._ndims):
            res.append(i + 1)
        return [res]

@tag(["VLSE", "Other", "Unconstrained", "Multimodal", "nD", "Continuous", "Differentiable", "Non_separable"])
class Powell(BenchmarkFunction):
    """
    The n Dimensional Powell function is multimodal. The last d mod 4 dimensions
    do not affect the function value.

    :Reference: https://www.sfu.ca/~ssurjano/michal.html
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

        for i in range(1, 1 + d // 4):
            res += (x[4*i - 4] + 10 * x[4*i - 3])**2 + 5 * (x[4*i - 2] - x[4*i - 1])**2 + (x[4*i - 3] - 2 * x[4*i - 2])**4 + 10 * (x[4*i - 4] - x[4*i - 1])**4
        return res

    def min(self):
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
        d = self._ndims
        return [[-4, 5] for _ in range(d)]

    def argmin(self):
        """
        Returns function argmin.

        :return: List of minimizer(s)
        """
        return [[0.0 for i in range(self._ndims)]]

@tag(["VLSE", "Other", "Unconstrained", "Multimodal", "4D", "Continuous", "Differentiable", "Non_separable"])
class Shekel(BenchmarkFunction):
    """
    The Shekel function with m = 10 is a 4D multimodal function with 10 local
    minima.

    :Reference: https://www.sfu.ca/~ssurjano/shekel.html
    """

    # Acceptable dimensions. Either integer or tuple.
    # If tuple, use -1 to show 'no upper bound'.
    DIM = 4

    def __init__(self, n: int = 4) -> None:
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

        m = 10

        beta = [0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5]

        C = [[4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0],
             [4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6],
             [4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0],
             [4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6]]

        res = 0.0
        for i in range(m):
            inner = 0.0
            for j in range(4):
                inner += (x[j] - C[j][i])**2
            res += 1 / (beta[i] + inner)

        return -res

    @staticmethod
    def min():
        """
        Returns known minimum function value.

        :return: Minimum value (float)
        """
        return -10.5364

    @staticmethod
    def bounds():
        """
        Returns problem bounds.

        :return: List of [lower, upper] for each dimension
        """
        return [[0, 10] for _ in range(4)]

    @staticmethod
    def argmin():
        """
        Returns function argmin.

        :return: List of minimizer(s)
        """
        return [[4, 4, 4, 4]]

@tag(["VLSE", "Other", "Unconstrained", "Multimodal", "nD", "Continuous", "Differentiable", "Separable"])
class StyblinskiTang(BenchmarkFunction):
    """
    The Styblinski-Tang function is a multimodal n Dimensional function.

    :Reference: https://www.sfu.ca/~ssurjano/stybtang.html
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

        for i in range(d):
            res += x[i]**4 - 16 * x[i]**2 + 5 * x[i]
        return 0.5 * res

    def min(self):
        """
        Returns known minimum function value.

        :return: Minimum value (float)
        """
        d = self._ndims
        return -39.16599 * d

    def bounds(self):
        """
        Returns problem bounds.

        :return: List of [lower, upper] for each dimension
        """
        d = self._ndims
        return [[-5, 5] for _ in range(d)]

    def argmin(self):
        """
        Returns function argmin.

        :return: List of minimizer(s)
        """
        return [[-2.903534 for _ in range(self._ndims)]]

# =============================================================================
# Constrained Benchmark Problems
# =============================================================================
@tag(["Constrained", "2D", "Continuous", "Differentiable", "Non_separable"])
class RosenbrockConstrained(BenchmarkFunction):
    """
    The Rosenbrock function constrained within and on the unit circle.

    Refernces:
        https://www.mathworks.com/help/optim/ug/example-nonlinear-constrained-minimization.html?w.mathworks.com=
    """

    # Acceptable dimensions. Either integer or tuple.
    # If tuple, use -1 to show 'no upper bound'.
    DIM = 2

    def __init__(self, n: int = 2) -> None:
        super().__init__(n)

    @staticmethod
    def evaluate(x, xp=None):
        """
        Evaluate the objective function at a given point.

        :param x: Input point (array-like)
        :param xp: Optional array API namespace (e.g., numpy, Torch)
        :return: Scalar function output
        """
        if xp is None:
            if isinstance(x, list):
                xp = np
            else:
                xp = array_api_compat.array_namespace(x)

        x1 = x[0]
        x2 = x[1]

        term1 = (1 - x1)**2
        term2 = 100 * (x2 - x1**2)**2
        return term1 + term2

    @staticmethod
    def constraint1(x, xp=None):
        """
        Evaluate the constraint function at a given point.
        Returns a negative value if the constraint is violated.

        :param x: Input point (array-like)
        :param xp: Optional array API namespace (e.g., numpy, Torch)
        :return: Scalar constraint output
        """
        if xp is None:
            if isinstance(x, list):
                xp = np
            else:
                xp = array_api_compat.array_namespace(x)

        x1 = x[0]
        x2 = x[1]
        return 1 - (x1**2 + x2**2)

    def constraints(self):
        """
        Returns the constraints of the problem.

        :return: List of constraint functions for this benchmark
        """
        return [self.constraint1]

    @staticmethod
    def min():
        """
        Returns known minimum function value.

        :return: Minimum value (float)
        """
        return 0.045678

    @staticmethod
    def bounds():
        """
        Returns problem bounds.

        :return: List of [lower, upper] for each dimension
        """
        return [[-1, 1], [-1, 1]]

    @staticmethod
    def argmin():
        """
        Returns function argmin.

        :return: List of minimizer(s)
        """
        return [[0.7864, 0.6177]]

@tag(["Constrained", "2D", "Continuous", "Differentiable", "Non_separable"])
class Bird(BenchmarkFunction):
    """
    The Bird Problem is a constrained problem with one global minimum
    and multiple local minima.

    References:
        https://web.archive.org/web/20161229032528/http://www.phoenix-int.com/software/benchmark_report/bird_constrained.php
    """

    # Acceptable dimensions. Either integer or tuple.
    # If tuple, use -1 to show 'no upper bound'.
    DIM = 2

    def __init__(self, n: int = 2) -> None:
        super().__init__(n)

    @staticmethod
    def evaluate(x, xp=None):
        """
        Evaluate the objective function at a given point.

        :param x: Input point (array-like)
        :param xp: Optional array API namespace (e.g., numpy, Torch)
        :return: Scalar function output
        """
        if xp is None:
            if isinstance(x, list):
                xp = np
            else:
                xp = array_api_compat.array_namespace(x)

        x1 = x[0]
        x2 = x[1]

        term1 = xp.sin(x1) * xp.exp((1-xp.cos(x2))**2)
        term2 = xp.cos(x2) * xp.exp((1-xp.sin(x1))**2)
        term3 = (x1 - x2)**2
        return term1 + term2 + term3

    @staticmethod
    def constraint1(x, xp=None):
        """
        Evaluate the constraint function at a given point.
        Returns a negative value if the constraint is violated.

        :param x: Input point (array-like)
        :param xp: Optional array API namespace (e.g., numpy, Torch)
        :return: Scalar constraint output
        """
        if xp is None:
            if isinstance(x, list):
                xp = np
            else:
                xp = array_api_compat.array_namespace(x)

        x1 = x[0]
        x2 = x[1]

        return (x1+5)**2 + (x2+5)**2 - 25

    def constraints(self):
        """
        Returns the constraints of the problem.

        :return: List of constraint functions for this benchmark
        """
        return [self.constraint1]

    @staticmethod
    def min():
        """
        Returns known minimum function value.

        :return: Minimum value (float)
        """
        return -106.764537

    @staticmethod
    def bounds():
        """
        Returns problem bounds.

        :return: List of [lower, upper] for each dimension
        """
        return [[-6, float('inf')], [-float('inf'), 6]]

    @staticmethod
    def argmin():
        """
        Returns function argmin.

        :return: List of minimizer(s)
        """
        return [[4.70104 ,3.15294]]

@tag(["Constrained", "4D", "Continuous", "Differentiable"])
class RosenSuzuki(BenchmarkFunction):
    """
    The Rosen-Suzuki function is a 4D constrained optimization problem
    with three constraints and one objective.
    It has a single global minimum.

    References:
        https://web.archive.org/web/20150406025243/http://www.phoenix-int.com:80/software/benchmark_report/rosen_suzuki.php
    """

    # Acceptable dimensions. Either integer or tuple.
    # If tuple, use -1 to show 'no upper bound'.
    DIM = 4

    def __init__(self, n: int = 4) -> None:
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

        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        x4 = x[3]

        return x1**2 + x2**2 + 2*x3**2 + x4**2 - 5*x1 -5*x2 - 21*x3 + 7*x4

    @staticmethod
    def constraint1(x, xp=None):
        """
        Evaluate the first constraint function at a given point.
        Returns a negative value if the constraint is violated.

        :param x: Input point (array-like)
        :param xp: Optional array API namespace (e.g., numpy, Torch)
        :return: Scalar constraint output
        """
        if xp is None:
            if isinstance(x, list):
                xp = np
            else:
                xp = array_api_compat.array_namespace(x)

        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        x4 = x[3]

        return -x1**2 - x2**2 - x3**2 - x4**2 - x1 + x2 - x3 + x4 + 8

    @staticmethod
    def constraint2(x, xp=None):
        """
        Evaluate the second constraint function at a given point.
        Returns a negative value if the constraint is violated.

        :param x: Input point (array-like)
        :param xp: Optional array API namespace (e.g., numpy, Torch)
        :return: Scalar constraint output
        """
        if xp is None:
            if isinstance(x, list):
                xp = np
            else:
                xp = array_api_compat.array_namespace(x)

        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        x4 = x[3]

        return -x1**2 - 2*x2**2 - x3**2 - 2*x4**2 + x1 + x4 + 10

    @staticmethod
    def constraint3(x, xp=None):
        """
        Evaluate the third constraint function at a given point.
        Returns a negative value if the constraint is violated.

        :param x: Input point (array-like)
        :param xp: Optional array API namespace (e.g., numpy, Torch)
        :return: Scalar constraint output
        """
        if xp is None:
            if isinstance(x, list):
                xp = np
            else:
                xp = array_api_compat.array_namespace(x)

        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        x4 = x[3]

        return -2*x1**2 - x2**2 - x3**2 - 2*x1 + x2 + x4 + 5

    def constraints(self):
        """
        Returns the constraints of the problem.

        :return: List of constraint functions for this benchmark
        """
        return [self.constraint1, self.constraint2, self.constraint3]

    @staticmethod
    def min():
        """
        Returns known minimum function value.

        :return: Minimum value (float)
        """
        return -44.0

    @staticmethod
    def bounds():
        """
        Returns problem bounds.

        :return: List of [lower, upper] for each dimension
        """
        return [[-3, 3] for _ in range(4)]

    @staticmethod
    def argmin():
        """
        Returns function argmin.

        :return: List of minimizer(s)
        """
        return [[0, 1, 2, -1]]
