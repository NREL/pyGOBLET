import numpy as np

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
__Scalable__ = []
__Non_scalable__ = []
__nD__ = []
__2D__ = []
__1D__ = []

def tag(tags):
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

# Template for new functions
class className:
    """
    Template for a benchmark function class.
    """

    # Acceptable dimensions. Either integer or tuple.
    # If tuple, use -1 to show 'no upper bound'.
    DIM = None

    def __init__(self, n: int) -> None:
        pass

    @staticmethod
    def evaluate(x):
        """
        Evaluate the function at a given point.

        :param x: Input point (array-like)
        :return: Scalar function output
        """

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

@tag(["Multimodal", "Continuous", "nD", "Differentiable", "Non_separable", "Scalable"])
class Ackley:
    """
    The Ackley function is a N-dimensional function with many local minima
    throughout the domain.

    Reference: https://www.sfu.ca/~ssurjano/ackley.html
    """

    # Acceptable dimensions. Either integer or tuple.
    # If tuple, use -1 to show 'no upper bound'.
    DIM = (1, -1)

    def __init__(self, n: int = DIM[0]) -> None:
        self._ndims = n

    @staticmethod
    def evaluate(x, a=20.0, b=0.2, c=2*np.pi):
        """
        Evaluate the Ackley function.

        :param x: n-d input point (array-like)
        :param a: float, default 20
        :param b: float, default 0.2
        :param c: float, default 2π
        :return: Scalar function output
        """
        x = np.asarray(x)
        d = len(x)

        # Compute the Ackley function
        term1 = -1 * a * np.exp(-1 * b * np.sqrt(np.sum(x**2) / d))
        term2 = -1 * np.exp(np.sum(np.cos(c * x)) / d)

        res = term1 + term2 + a + np.exp(1)

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

@tag(["Multimodal", "2D", "Continuous", "Non_differentiable", "Non_separable", "Non_scalable"])
class Bukin6:
    """
    Bukin Function N. 6 is a 2D function with many local minima along a ridge.

    Reference: https://www.sfu.ca/~ssurjano/bukin6.html
    """

    # Acceptable dimensions. Either integer or tuple.
    # If tuple, use -1 to show 'no upper bound'.
    DIM = 2

    def __init__(self, n: int) -> None:
        pass

    @staticmethod
    def evaluate(x):
        """
        Evaluate the Bukin6 function.

        :param x: 2D input point (array-like)
        :return: Scalar function output
        """
        # Unpack the input vector
        x1, x2 = x

        # Compute the Bukin function
        term1 = 100 * np.sqrt(np.abs(x2 - 0.01 * x1**2))
        term2 = 0.01 * np.abs(x1 + 10)

        res = term1 + term2

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
        return [[-15, -5], [-3, 3]]

    @staticmethod
    def argmin():
        """
        Returns function argmin.

        :return: List of minimizer(s)
        """
        return [-10, 1]

@tag(["Multimodal", "2D", "Continuous", "Non_separable", "Non_scalable"])
class CrossInTray:
    """
    The Cross-in-Tray is a 2D function with many local minima and
    four global minima.

    Reference: http://infinity77.net/global_optimization/test_functions_nd_C.html
    """

    # Acceptable dimensions. Either integer or tuple.
    # If tuple, use -1 to show 'no upper bound'.
    DIM = 2

    def __init__(self, n: int) -> None:
        pass

    @staticmethod
    def evaluate(x):
        """
        Evaluate the Cross-in-Tray function.

        :param x: 2D input point (array-like)
        :return: Scalar function output
        """
        # Unpack the input vector
        x1, x2 = x

        # Compute the Cross-in-Tray function
        term1 = np.abs(np.sin(x1) * np.sin(x2))
        term2 = np.exp(np.abs(100 - np.sqrt(x1**2 + x2**2) / np.pi))

        res = -0.0001 * (term1 * term2)

        return res

    @staticmethod
    def min():
        """
        Returns known minimum function value.

        :return: Minimum value (float)
        """
        return 2.062611870822739

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

@tag(["Multimodal", "2D", "Continuous"])
class DropWave:
    """
    The Drop-Wave function is a multimodal 2D function with many local minima
    and one global minimum.

    Reference: https://www.sfu.ca/~ssurjano/drop.html
    """

    # Acceptable dimensions. Either integer or tuple.
    # If tuple, use -1 to show 'no upper bound'.
    DIM = 2

    def __init__(self, n: int) -> None:
        pass

    @staticmethod
    def evaluate(x):
        """
        Evaluate the Drop-Wave function.

        :param x: 2D input point (array-like)
        :return: Scalar function output
        """
        # Unpack the input vector
        x1, x2 = x

        # Compute the Drop-Wave function
        num = 1 + np.cos(12 * np.sqrt(x1**2 + x2**2))
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

@tag(["Multimodal", "2D", "Continuous", "Differentiable", "Non_separable", "Scalable"])
class EggHolder:
    """
    The Eggholder function is a 2D function with many local minima and
    one global minimum.

    Reference: http://infinity77.net/global_optimization/test_functions_nd_E.html
    """

    # Acceptable dimensions. Either integer or tuple.
    # If tuple, use -1 to show 'no upper bound'.
    DIM = 2

    def __init__(self, n: int) -> None:
        pass

    @staticmethod
    def evaluate(x):
        """
        Evaluate the Eggholder function.

        :param x: 2D input point (array-like)
        :return: Scalar function output
        """
        # Unpack the input vector
        x1, x2 = x

        # Compute the Eggholder function
        term1 = -(x2 + 47) * np.sin(np.sqrt(np.abs(x1 / 2 + x2 + 47)))
        term2 = -x1 * np.sin(np.sqrt(np.abs(x1 - (x2 + 47))))

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
        return [512, 404.2319]

@tag(["Multimodal", "1D", "Continuous", "Differentiable"])
class GramacyLee:
    """
    The Gramacy-Lee function is a 1D function with many local minima
    and one global minimum.

    Reference: https://www.sfu.ca/~ssurjano/grlee12.html
    """

    # Acceptable dimensions. Either integer or tuple.
    # If tuple, use -1 to show 'no upper bound'.
    DIM = 1

    def __init__(self, n: int) -> None:
        pass

    @staticmethod
    def evaluate(x):
        """
        Evaluate the Gramacy-Lee function.

        :param x: 1D input point
        :return: Scalar function output
        """
        # Compute the Gramacy-Lee function
        term1 = np.sin(10 * np.pi * x) / (2 * x)
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
        return 0.548563444114526

@tag(["Multimodal", "nD", "Continuous", "Differentiable", "Non_separable", "Scalable"])
class Griewank:
    """
    The Griewank function is a N-dimensional function with many local minima
    and one global minimum.

    Reference: https://www.sfu.ca/~ssurjano/griewank.html
    """

    # Acceptable dimensions. Either integer or tuple.
    # If tuple, use -1 to show 'no upper bound'.
    DIM = (1, -1)

    def __init__(self, n: int = DIM[0]) -> None:
        self._ndims = n

    @staticmethod
    def evaluate(x):
        """
        Evaluate the Griewank function.

        :param x: n-d input point (array-like)
        :return: Scalar function output
        """
        return 1 + np.sum(x**2) / 4000 - np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))

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

@tag(["Multimodal", "2D", "Continuous", "Differentiable", "Separable", "Non_scalable"])
class HolderTable:
    """
    The Holder Table function is a 2D function with many local minima
    and four global minima.

    Reference: https://www.sfu.ca/~ssurjano/holder.html
    """

    # Acceptable dimensions. Either integer or tuple.
    # If tuple, use -1 to show 'no upper bound'.
    DIM = 2

    def __init__(self, n: int) -> None:
        pass

    @staticmethod
    def evaluate(x):
        """
        Evaluate the Holder Table function.

        :param x: 2D input point (array-like)
        :return: Scalar function output
        """
        # Unpack the input vector
        x1, x2 = x

        # Compute the Holder Table function
        term1 = np.sin(x1) * np.cos(x2)
        term2 = np.exp(np.abs(1 - (np.sqrt(x1**2 + x2**2) / np.pi)))

        res = -np.abs(term1 * term2)

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

@tag(["Multimodal", "2D", "Continuous", "Differentiable", "Non_separable", "Scalable"])
class Langermann:
    """
    The Langermann function is a 2D function with many local minima and
    one global minimum.

    Reference: https://www.sfu.ca/~ssurjano/langer.html
    https://infinity77.net/global_optimization/test_functions_nd_L.html#go_benchmark.Langermann
    """

    # Acceptable dimensions. Either integer or tuple.
    # If tuple, use -1 to show 'no upper bound'.
    DIM = 2

    def __init__(self, n: int) -> None:
        pass

    @staticmethod
    def evaluate(x):
        """
        Evaluate the Langermann function.

        :param x: 2D input point (array-like)
        :return: Scalar function output
        """
        A = np.array([[3, 5], [5, 2], [2, 1], [1, 4], [7, 9]])
        c = np.array([1, 2, 5, 2, 3])
        x = np.asarray(x)

        x = np.asarray(x)
        m, d = A.shape
        xxmat = np.tile(x, (m, 1))
        inner = np.sum((xxmat - A[:, :d]) ** 2, axis=1)
        res = -np.sum(c * np.exp(-inner / np.pi) * np.cos(np.pi * inner))
        return res

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

@tag(["Multimodal", "nD", "Continuous"])
class Levy:
    """
    The Levy Function is a N-dimensional function with many local minima and
    one global minimum.

    Reference: https://www.sfu.ca/~ssurjano/levy.html
    """

    # Acceptable dimensions. Either integer or tuple.
    # If tuple, use -1 to show 'no upper bound'.
    DIM = (1, -1)

    def __init__(self, n: int = DIM[0]) -> None:
        self._ndims = n

    @staticmethod
    def evaluate(x):
        """
        Evaluate the Levy function.

        :param x: n-d input point (array-like)
        :return: Scalar function output
        """
        x = np.asarray(x)
        w = 1 + (x - 1) / 4

        # Compute the Levy function
        term1 = np.sin(np.pi * w[0])**2
        term2 = np.sum((w[:-1] - 1)**2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1)**2))
        term3 = (w[-1] - 1)**2 * (1 + np.sin(2 * np.pi * w[-1])**2)

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

@tag(["Multimodal", "2D", "Continuous"])
class Levy13:
    """
    Levy 13 is a 2D function with many local minima and one global minimum.

    Reference: https://www.sfu.ca/~ssurjano/levy13.html
    """

    # Acceptable dimensions. Either integer or tuple.
    # If tuple, use -1 to show 'no upper bound'.
    DIM = 2

    def __init__(self, n: int) -> None:
        pass

    @staticmethod
    def evaluate(x):
        """
        Evaluate the Levy 13 function.

        :param x: 2D input point (array-like)
        :return: Scalar function output
        """
        # Unpack the input vector
        x1, x2 = x

        # Compute the Levy function
        term1 = np.sin(3 * np.pi * x1)**2
        term2 = (x1 - 1)**2 * (1 + np.sin(3 * np.pi * x2)**2)
        term3 = (x2 - 1)**2 * (1 + np.sin(2 * np.pi * x2)**2)

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

@tag(["Multimodal", "nD", "Continuous"])
class Rastrigin:
    """
    The Rastrigin function is a N-dimensional function with many local minima
    and one global minimum.

    Reference: https://www.sfu.ca/~ssurjano/rastr.html
    """

    # Acceptable dimensions. Either integer or tuple.
    # If tuple, use -1 to show 'no upper bound'.
    DIM = (1, -1)

    def __init__(self, n: int = DIM[0]) -> None:
        self._ndims = n

    @staticmethod
    def evaluate(x):
        """
        Evaluate the Rastrigin function.

        :param x: n-d input point (array-like)
        :return: Scalar function output
        """
        x = np.asarray(x)
        d = len(x)

        # Compute the Rastrigin function
        term1 = 10 * d
        term2 = np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

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

@tag(["Multimodal", "2D", "Continuous"])
class Schaffer2:
    """
    The second Schaffer function is a 2D function with many local minima and
    one global minimum.

    Reference: https://www.sfu.ca/~ssurjano/schaffer2.html
    """

    # Acceptable dimensions. Either integer or tuple.
    # If tuple, use -1 to show 'no upper bound'.
    DIM = 2

    def __init__(self, n: int) -> None:
        pass

    @staticmethod
    def evaluate(x):
        """
        Evaluate the Schaffer 2 function.

        :param x: 2D input point (array-like)
        :return: Scalar function output
        """
        # Unpack the input vector
        x1, x2 = x

        # Compute the Schaffer function
        numer = np.sin(x1**2 - x2**2)**2 - 0.5
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

@tag(["Multimodal", "2D", "Continuous"])
class Schaffer4:
    """
    The fourth Schaffer function is a 2D function with many local minima and
    four global minima.

    Reference: https://www.sfu.ca/~ssurjano/schaffer4.html
    https://en.wikipedia.org/wiki/Test_functions_for_optimization
    Mishra, S. Some new test functions for global optimization and
    performance of repulsive particle swarm method. Munich Personal
    RePEc Archive, 2006, 2718
    """

    # Acceptable dimensions. Either integer or tuple.
    # If tuple, use -1 to show 'no upper bound'.
    DIM = 2

    def __init__(self, n: int) -> None:
        pass

    @staticmethod
    def evaluate(x):
        """
        Evaluate the Schaffer 4 function.

        :param x: 2D input point (array-like)
        :return: Scalar function output
        """
        # Unpack the input vector
        x1, x2 = x

        # Compute the Schaffer function
        numer = np.cos( np.sin( np.abs( x1**2 - x2**2 ) ) )**2 - 0.5
        denom = ( 1 + 0.001 * (x1**2 + x2**2) )**2

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

@tag(["Multimodal", "nD", "Continuous", "Differentiable", "Separable", "Scalable"])
class Schwefel:
    """
    The Schwefel function is a N-dimensional function with many local minima and
    one global minimum.

    Reference: https://www.sfu.ca/~ssurjano/schwef.html
    """

    # Acceptable dimensions. Either integer or tuple.
    # If tuple, use -1 to show 'no upper bound'.
    DIM = (1, -1)

    def __init__(self, n: int = DIM[0]) -> None:
        self._ndims = n

    @staticmethod
    def evaluate(x):
        """
        Evaluate the Schwefel function.

        :param x: n-d input point (array-like)
        :return: Scalar function output
        """
        x = np.asarray(x)
        d = len(x)

        # Compute the Schwefel function
        term1 = 418.9829 * d
        term2 = np.sum(x * np.sin(np.sqrt(np.abs(x))))

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

@tag(["Multimodal", "2D", "Continuous", "Differentiable", "Non_scalable"])
class Shubert:
    """
    The Shubert function is a 2D function with many local minima and
    18 Global minima.

    Reference: https://www.sfu.ca/~ssurjano/shubert.html
    """

    # Acceptable dimensions. Either integer or tuple.
    # If tuple, use -1 to show 'no upper bound'.
    DIM = 2

    def __init__(self, n: int) -> None:
        pass

    @staticmethod
    def evaluate(x):
        """
        Evaluate the Shubert function.

        :param x: 2D input point (array-like)
        :return: Scalar function output
        """
        # Unpack the input vector
        x1, x2 = x

        # Compute the Shubert function
        term1 = np.sum([i * np.cos((i + 1) * x1 + i) for i in range(1, 6)])
        term2 = np.sum([i * np.cos((i + 1) * x2 + i) for i in range(1, 6)])

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

@tag(["Multimodal", "2D", "Continuous", "Differentiable", "Separable", "Non_scalable"])
class Bohachevsky1:
    """
    The Bohachevsky functions are bowl-shaped.

    Reference: https://www.sfu.ca/~ssurjano/boha.html
    """

    # Acceptable dimensions. Either integer or tuple.
    # If tuple, use -1 to show 'no upper bound'.
    DIM = 2

    def __init__(self, n: int) -> None:
        pass

    @staticmethod
    def evaluate(x):
        """
        Evaluate the Bohachevsky 1 function.

        :param x: 2D input point (array-like)
        :return: Scalar function output
        """
        # Unpack the input vector
        x1, x2 = x

        # Compute the Bohachevsky function
        term1 = x1**2 + 2 * x2**2
        term2 = 0.3 * np.cos(3 * np.pi * x1) + 0.4 * np.cos(4 * np.pi * x2)

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

@tag(["Multimodal", "2D", "Continuous", "Differentiable", "Non_separable", "Non_scalable"])
class Bohachevsky2:
    """
    The Bohachevsky functions are bowl-shaped.

    Reference: https://www.sfu.ca/~ssurjano/boha.html
    """

    # Acceptable dimensions. Either integer or tuple.
    # If tuple, use -1 to show 'no upper bound'.
    DIM = 2

    def __init__(self, n: int) -> None:
        pass

    @staticmethod
    def evaluate(x):
        """
        Evaluate the Bohachevsky 2 function.

        :param x: 2D input point (array-like)
        :return: Scalar function output
        """
        # Unpack the input vector
        x1, x2 = x

        # Compute the Bohachevsky function
        term1 = x1**2 + 2 * x2**2
        term2 = 0.3 * np.cos(3 * np.pi * x1) * np.cos(4 * np.pi * x2)

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

@tag(["Multimodal", "2D", "Continuous", "Differentiable", "Non_separable", "Non_scalable"])
class Bohachevsky3:
    """
    The Bohachevsky functions are bowl-shaped.

    Reference: https://www.sfu.ca/~ssurjano/boha.html
    """

    # Acceptable dimensions. Either integer or tuple.
    # If tuple, use -1 to show 'no upper bound'.
    DIM = 2

    def __init__(self, n: int) -> None:
        pass

    @staticmethod
    def evaluate(x):
        """
        Evaluate the Bohachevsky 3 function.

        :param x: 2D input point (array-like)
        :return: Scalar function output
        """
        # Unpack the input vector
        x1, x2 = x

        # Compute the Bohachevsky function
        term1 = x1**2 + 2 * x2**2
        term2 = 0.3 * np.cos(3 * np.pi * x1 + 4 * np.pi * x2)

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

## IMPLEMENTATION INCOMPLETE
class Perm:
    """
    The Perm function is a N-dimensional bowl-shaped function.

    Reference: https://www.sfu.ca/~ssurjano/perm0db.html
    """

    # Acceptable dimensions. Either integer or tuple.
    # If tuple, use -1 to show 'no upper bound'.
    DIM = (1, -1)

    def __init__(self, n: int = DIM[0]) -> None:
        self._ndims = n

    @staticmethod
    def evaluate(x, beta=10.0):
        """
        Evaluate the Perm function.

        :param x: 2D input point (array-like)
        :param beta: float, default 10.0
        :return: Scalar function output
        """
        # d = len(x)

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
        return [list( 1/ np.arange(1, self._ndims + 1) )]
