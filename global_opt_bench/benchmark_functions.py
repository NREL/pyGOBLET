"""
10/16 Many Local Minima Functions from https://www.sfu.ca/~ssurjano/optimization.html

"""
import numpy as np

# All available test functions
# TODO: update tagging system to allow each function to be tagged with multiple categories
__All__ = [
    "Ackley",
    "Bukin6",
    "ClassInTray",
    "DropWave",
    "EggHolder",
    "GramacyLee",
    "Griewank",
    "HolderTable",
    "Levy",
    "Levy13",
]

# Template for new functions
class className:

    # Acceptable dimensions. Either integer or tuple.
    # If tuple, use -1 to show 'no upper bound'.
    DIM = None

    def __init__(self, n: int) -> None:
        pass

    @staticmethod
    def evaluate(x):
        """Explanation

        Reference:
        
        Parameters
        ----------
        Param1
            info

        Returns
        -------
        res
            info
            
        """

    @staticmethod
    def min():
        """Returns known minimum function value"""

    @staticmethod
    def bounds():
        """Returns problem bounds"""

    @staticmethod
    def argmin():
        """Returns function argmin"""

class Ackley:

    # Acceptable dimensions. Either integer or tuple.
    # If tuple, use -1 to show 'no upper bound'.
    DIM = (1, -1)

    def __init__(self, n: int = DIM[0]) -> None:
        self._ndims = n

    @staticmethod
    def evaluate(x, a=20.0, b=0.2, c=2*np.pi):
        """The Ackley function is a N-dimensional function with many local minima throughout the domain.

        Reference: https://www.sfu.ca/~ssurjano/ackley.html
        
        Parameters
        ----------
        x 
            N-d input point
        a : float
            Default value 20
        b : float
            Default value 0.2
        c : float 
            Default value 2π
        
        Returns
        -------
        res : float
            Scalar function output
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
        """Returns known minimum function value"""

        return 0.0

    def bounds(self):
        """Returns problem bounds"""

        return [[-32.768, 32.768] for i in range(self._ndims)]

    def argmin(self):
        """Returns function argmin"""
        return [[0.0] for i in range(self._ndims)]

class Bukin6:

    # Acceptable dimensions. Either integer or tuple.
    # If tuple, use -1 to show 'no upper bound'.
    DIM = 2

    def __init__(self, n: int) -> None:
        pass
    
    @staticmethod
    def evaluate(x):
        """Bukin Function N. 6 is a 2D funciton with many local minima which all are along a ridge.

        Reference: https://www.sfu.ca/~ssurjano/bukin6.html
        
        Parameters
        ----------
        x
            2D input point

        Returns
        -------
        res : float
            Scalar function output
            
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
        """Returns known minimum function value"""
        return 0.0

    @staticmethod
    def bounds():
        """Returns problem bounds"""
        return [[-15, -5], [-3, 3]]
    
    @staticmethod
    def argmin():
        """Returns function argmin"""
        return [-10, 1]
    
class ClassInTray:

    # Acceptable dimensions. Either integer or tuple.
    # If tuple, use -1 to show 'no upper bound'.
    DIM = 2

    def __init__(self, n: int) -> None:
        pass

    @staticmethod
    def evaluate(x):
        """The Cross-in-Tray is a 2D function has many local minima and four global minima.

        Reference: http://infinity77.net/global_optimization/test_functions_nd_C.html 
        
        Parameters
        ----------
        x
            2D input point

        Returns
        -------
        res : float
            Scalar function output
            
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
        """Returns known minimum function value"""
        return 2.062611870822739

    @staticmethod
    def bounds():
        """Returns problem bounds"""
        return [[-10, 10], [-10, 10]]

    @staticmethod
    def argmin():
        """Returns function argmin"""
        return [[1.349406608602084, -1.349406608602084], [1.349406608602084, 1.349406608602084], [-1.349406608602084, 1.349406608602084], [-1.349406608602084, -1.349406608602084]]

class DropWave:

    # Acceptable dimensions. Either integer or tuple.
    # If tuple, use -1 to show 'no upper bound'.
    DIM = 2

    def __init__(self, n: int) -> None:
        pass

    @staticmethod
    def evaluate(x):
        """The Drop-Wave function is a multimodal 2D function with many local minima and one global minimum.

        Reference: https://www.sfu.ca/~ssurjano/drop.html
        
        Parameters
        ----------
        x
            2D input point

        Returns
        -------
        res : float
            Scalar function output
            
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
        """Returns known minimum function value"""
        return -1.0

    @staticmethod
    def bounds():
        """Returns problem bounds"""
        return [[-5.12, 5.12], [-5.12, 5.12]]

    @staticmethod
    def argmin():
        """Returns function argmin"""
        return [[0.0, 0.0]]
    
class EggHolder:

    # Acceptable dimensions. Either integer or tuple.
    # If tuple, use -1 to show 'no upper bound'.
    DIM = 2

    def __init__(self, n: int) -> None:
        pass

    @staticmethod
    def evaluate(x):
        """The Eggholder function is a 2D function with many local minima and one global minimum.

        Reference: http://infinity77.net/global_optimization/test_functions_nd_E.html
        
        Parameters
        ----------
        x
            2D input point

        Returns
        -------
        res : float
            Scalar function output
            
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
        """Returns known minimum function value"""
        return -959.640662711

    @staticmethod
    def bounds():
        """Returns problem bounds"""
        return [[-512, 512], [-512, 512]]

    @staticmethod
    def argmin():
        """Returns function argmin"""
        return [512, 404.2319]

class GramacyLee:

    # Acceptable dimensions. Either integer or tuple.
    # If tuple, use -1 to show 'no upper bound'.
    DIM = 1

    def __init__(self, n: int) -> None:
        pass

    @staticmethod
    def evaluate(x):
        """The Gramacy-Lee function is a 1D function with many local minima and one global minimum.

        Reference: https://www.sfu.ca/~ssurjano/grlee12.html
        
        Parameters
        ----------
        x : float
            1D input point

        Returns
        -------
        res : float
            Scalar function output
            
        """
        # Compute the Gramacy-Lee function
        term1 = np.sin(10 * np.pi * x) / (2 * x)
        term2 = (x - 1)**4

        res = term1 + term2

        return res

    @staticmethod
    def min():
        """Returns known minimum function value"""
        return -0.869011134989500

    @staticmethod
    def bounds():
        """Returns problem bounds"""
        return [[0.5, 2.5]]

    @staticmethod
    def argmin():
        """Returns function argmin"""
        return 0.548563444114526

class Griewank:

    # Acceptable dimensions. Either integer or tuple.
    # If tuple, use -1 to show 'no upper bound'.
    DIM = (1, -1)

    def __init__(self, n: int = DIM[0]) -> None:
        self._ndims = n

    @staticmethod
    def evaluate(x):
        """The Griewank function is a N-dimensional function with many local minima and one global minimum.

        Reference: https://www.sfu.ca/~ssurjano/griewank.html
        
        Parameters
        ----------
        x
            N-d input point

        Returns
        -------
        res : float
            Scalar function output
            
        """
        return 1 + np.sum(x**2) / 4000 - np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))

    @staticmethod
    def min():
        """Returns known minimum function value"""
        return 0.0

    def bounds(self):
        """Returns problem bounds"""
        return [[-600, 600] for i in range(self._ndims)]

    def argmin(self):
        """Returns function argmin"""
        return [[0.0] for i in range(self._ndims)]

class HolderTable:

    # Acceptable dimensions. Either integer or tuple.
    # If tuple, use -1 to show 'no upper bound'.
    DIM = 2

    def __init__(self, n: int) -> None:
        pass

    @staticmethod
    def evaluate(x):
        """The Holder Table function is a 2D function with many local minima and four global minima.

        Reference: https://www.sfu.ca/~ssurjano/holder.html
        
        Parameters
        ----------
        x
            2D input point

        Returns
        -------
        res : float
            Scalar function output
            
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
        """Returns known minimum function value"""
        return -19.2085

    @staticmethod
    def bounds():
        """Returns problem bounds"""
        return [[-10.0, 10.0], [-10.0, 10.0]]

    @staticmethod
    def argmin():
        """Returns function argmin"""
        return [[8.05502, 9.66459], [-8.05502, -9.66459], [8.05502, -9.66459], [-8.05502, 9.66459]]

## Problem Child
class Langermann:

    # Acceptable dimensions. Either integer or tuple.
    # If tuple, use -1 to show 'no upper bound'.
    DIM = 2

    def __init__(self, n: int) -> None:
        pass

    @staticmethod
    def evaluate(x):
        """The Langermann function is a 2D function with many
        local minima and one global minimum.

        Reference: https://www.sfu.ca/~ssurjano/langer.html
        
        Parameters
        ----------
        x
            2D input point

        Returns
        -------
        res : float
            Scalar function output
            
        """
        A = np.array([[3, 5], [5, 2], [2, 1], [1, 4], [7, 9]])
        c = np.array([1, 2, 5, 2, 3])
        x = np.asarray(x)



        
    @staticmethod
    def min():
        """Returns known minimum function value"""
        return -5.1621259

    @staticmethod
    def bounds():
        """Returns problem bounds"""
        return [[0.0, 10.0], [0.0, 10.0]]
    
    @staticmethod
    def argmin():
        """Returns function argmin"""
        return [[2.002992, 1.006096]]

class Levy:

    # Acceptable dimensions. Either integer or tuple.
    # If tuple, use -1 to show 'no upper bound'.
    DIM = (1, -1)

    def __init__(self, n: int = DIM[0]) -> None:
        self._ndims = n

    @staticmethod
    def evaluate(x):
        """The Levy Function is a N-dimensional function with many
        local minima and one global minimum.

        Reference: https://www.sfu.ca/~ssurjano/levy.html
        
        Parameters
        ----------
        x
            n-d input point

        Returns
        -------
        res : float
            Scalar function output
            
        """
        x = np.asarray(x)
        w = 1 + (x - 1) / 4
        d = len(w)

        # Compute the Levy function
        term1 = np.sin(np.pi * w[0])**2
        term2 = np.sum((w[:-1] - 1)**2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1)**2))
        term3 = (w[-1] - 1)**2 * (1 + np.sin(2 * np.pi * w[-1])**2)

        res = term1 + term2 + term3

        return res

    @staticmethod
    def min():
        """Returns known minimum function value"""
        return 0.0

    def bounds(self):
        """Returns problem bounds"""
        return [[-10.0, 10.0] for i in range(self._ndims)]

    def argmin(self):
        """Returns function argmin"""
        return [[1.0] for i in range(self._ndims)]

class Levy13:

    # Acceptable dimensions. Either integer or tuple.
    # If tuple, use -1 to show 'no upper bound'.
    DIM = 2

    def __init__(self, n: int) -> None:
        pass

    @staticmethod
    def evaluate(x):
        """Levy 13 is a 2D function with many local minima and one global minimum.

        Reference: https://www.sfu.ca/~ssurjano/levy13.html
        
        Parameters
        ----------
        x
            2D input point

        Returns
        -------
        res : float
            Scalar function output
            
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
        """Returns known minimum function value"""
        return 0.0

    @staticmethod
    def bounds():
        """Returns problem bounds"""
        return [[-10.0, 10.0], [-10.0, 10.0]]

    @staticmethod
    def argmin():
        """Returns function argmin"""
        return [[1.0, 1.0]]
