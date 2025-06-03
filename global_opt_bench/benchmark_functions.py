import numpy as np
import array_api_compat
try:
    import pyomo.environ as pyo
except ImportError:
    pyo = None

# Function categories
__All__ = []
__Multimodal__ = []
__Unimodal__ = []
__Continuous__ = []
__Discontinuous__ = []
__Differentiable__ = []
__Non_differentiable__ = []
__Separable__ = []
__Non_separable__ = []
__nD__ = []
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
        # Add to global registry
        if not hasattr(__import__(__name__), "__All__"):
            setattr(__import__(__name__), "__All__", [])
        __import__(__name__).__All__.append(cls)
        # Add to tag-specific lists
        for t in tags:
            if not hasattr(__import__(__name__), f"__{t}__"):
                setattr(__import__(__name__), f"__{t}__", [])
            getattr(__import__(__name__), f"__{t}__").append(cls)
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
                    const = c(model.x, xp=pyo)
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

# Template for new functions
class className(BenchmarkFunction):
    """
    Template for a benchmark function class.
    """

    # Acceptable dimensions. Either integer or tuple.
    # If tuple, use -1 to show 'no upper bound'.
    DIM = None

    def __init__(self, n: int) -> None:
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
            xp = array_api_compat.array_namespace(x)

    @staticmethod
    def min():
        """
        Returns known minimum function value.

        :return: Minimum value (float)
        """

    @staticmethod
    def bounds():
        """
        Returns problem bounds.

        :return: List of [lower, upper] for each dimension
        """

    @staticmethod
    def argmin():
        """
        Returns function argmin.

        :return: List of minimizer(s)
        """

@tag(["Unconstrained", "Multimodal", "Continuous", "nD", "Differentiable", "Non_separable"])
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
        return [[0.0] for i in range(self._ndims)]

@tag(["Unconstrained", "Multimodal", "2D", "Continuous", "Non_differentiable", "Non_separable"])
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

@tag(["Unconstrained", "Multimodal", "2D", "Continuous", "Non_separable"])
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

@tag(["Unconstrained", "Multimodal", "2D", "Continuous"])
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

@tag(["Unconstrained", "Multimodal", "2D", "Continuous", "Differentiable", "Non_separable"])
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

@tag(["Unconstrained", "Multimodal", "1D", "Continuous", "Differentiable"])
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

@tag(["Unconstrained", "Multimodal", "nD", "Continuous", "Differentiable", "Non_separable"])
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
        return [[0.0] for i in range(self._ndims)]

@tag(["Unconstrained", "Multimodal", "2D", "Continuous", "Differentiable", "Separable"])
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

@tag(["Unconstrained", "Multimodal", "2D", "Continuous", "Differentiable", "Non_separable"])
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

@tag(["Unconstrained", "Multimodal", "nD", "Continuous"])
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
        return [[1.0] for i in range(self._ndims)]

@tag(["Unconstrained", "Multimodal", "2D", "Continuous"])
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

@tag(["Unconstrained", "Multimodal", "nD", "Continuous"])
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
        return [[0.0] for i in range(self._ndims)]

@tag(["Unconstrained", "Multimodal", "2D", "Continuous"])
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

@tag(["Unconstrained", "Multimodal", "2D", "Continuous"])
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

@tag(["Unconstrained", "Multimodal", "nD", "Continuous", "Differentiable", "Separable"])
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
        return [[420.968746] for i in range(self._ndims)]

@tag(["Unconstrained", "Multimodal", "2D", "Continuous", "Differentiable"])
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

@tag(["Unconstrained", "Multimodal", "2D", "Continuous", "Differentiable", "Separable"])
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

@tag(["Unconstrained", "Multimodal", "2D", "Continuous", "Differentiable", "Non_separable"])
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

@tag(["Unconstrained", "Multimodal", "2D", "Continuous", "Differentiable", "Non_separable"])
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

# Constrained problem:
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
            xp = array_api_compat.array_namespace(x)

        x1 = x[0]
        x2 = x[1]
        if pyo is not None and xp == pyo:
            return x1**2 + x2**2 <= 1
        else:
            return 1 - x1**2 + x2**2

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
            xp = array_api_compat.array_namespace(x)

        x1 = x[0]
        x2 = x[1]
        if pyo is not None and xp == pyo:
            return (x1+5)**2 + (x2+5)**2 >= 25
        else:
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
