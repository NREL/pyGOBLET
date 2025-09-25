# Copyright (c) 2025 Alliance for Sustainable Energy, LLC

from abc import ABC, abstractmethod
from typing import Callable, List, Tuple, Optional, Union
import numpy as np

class OptimizationResult:
    """
    Standardized result object for all algorithms. Defines as
    attributes:

    * x: Solution vector (list, array-like)
    * fun: Objective function value at x (float, None)
    * algorithm: Name/identifier of the algorithm used (str)
    """
    def __init__(self, x, fun: Optional[float] = None, algorithm: str = ""):
        """
        :param x: Solution vector (list, array-like)
        :param fun: Objective function value at x
        :param algorithm: Name/identifier of the algorithm used
        """
        self.x = x
        self.fun = float(fun) if fun is not None else None
        self.algorithm = str(algorithm)

class BaseOptimizer(ABC):
    """
    Abstract base class for optimization algorithms tested with pyGOLD.

    This class defines the interface that all optimization algorithms must
    implement to be compatible with pyGOLD benchmark testing. It enforces a
    standardized approach to handling different algorithm types and their
    initialization requirements.

    All subclasses must implement the ``optimize`` method, which is the
    core optimization routine that takes an objective function, bounds,
    initial conditions, and constraints.

    All subclasses must define the following attributes:

    - ``deterministic`` (bool):
        True if the algorithm is deterministic, False if stochastic.
    - ``n_points`` (int):
        Number of initial points the algorithm requires.
    """
    deterministic: bool

    # n_points = <int>
    # 0: Algorithm requires no initial point
    # 1: Algorithm requires a single initial point
    # >1: Algorithm requires a set of initial points
    n_points: int

    def __init_subclass__(cls, **kwargs):
        """
        Ensure that subclasses define deterministic (bool) and
        n_points (int) attributes.
        If n_points is 0, the algorithm does not require initial conditions.
        If n_points is 1, the algorithm requires a single initial point.
        If n_points > 1, the algorithm requires a set of initial points.
        """
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, "deterministic"):
            raise TypeError(f"{cls.__name__} must define a 'deterministic' class attribute (True or False).")
        if not isinstance(cls.deterministic, bool):
            raise TypeError(f"{cls.__name__}.deterministic must be a bool (True for deterministic, False for stochastic).")
        if not hasattr(cls, "n_points"):
            raise TypeError(f"{cls.__name__} must define an 'n_points' class attribute (int).")
        if not isinstance(cls.n_points, int) or cls.n_points < 0:
            raise TypeError(f"{cls.__name__}.n_points must be a non-negative integer.")

    def __init__(self, **kwargs):
        """
        Initialize the optimizer.

        :param kwargs: Algorithm-specific parameters
        """
        self.kwargs = kwargs
        self.name = self.__class__.__name__

    @abstractmethod
    def optimize(self, func: Callable, bounds: List[Tuple[float, float]], x0: Optional[Union[np.ndarray, List[np.ndarray]]] = None, constraints: Optional[List[Callable]] = None, **kwargs) -> OptimizationResult:
        """
        Optimize the given function within specified bounds.

        :param func: Objective function to minimize. Should accept a numpy
            array and return a scalar float.
        :type func: Callable
        :param bounds: List of (min, max) tuples specifying the bounds for
            each dimension.
        :type bounds: List[Tuple[float, float]]
        :param x0: Initial condition(s). Type depends on n_points:

            * n_points == 0: Should be None (ignored if provided)
            * n_points == 1: Single initial point as array-like
            * n_points > 1: List of initial points as arrays

        :type x0: Optional[Union[np.ndarray, List[np.ndarray]]]
        :param constraints: List of constraint functions. Empty list if the
            problem is unconstrained. Constraint functions will return a
            negative value if violated.
        :type constraints: Optional[List[Callable]]
        :param kwargs: Additional algorithm-specific parameters.
        :type kwargs: dict

        :returns: The result of the optimization process.
        :rtype: OptimizationResult
        """
        pass
