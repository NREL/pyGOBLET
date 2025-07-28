from abc import ABC, abstractmethod
from typing import Callable, List, Tuple, Optional, Union
import numpy as np
from enum import Enum

class InitializationType(Enum):
    """Types of initialization requirements for optimization algorithms

    - NONE: Algorithm doesn't use initial conditions
    - SINGLE: Algorithm requires one initial point
    - MULTIPLE: Algorithm requires multiple initial points
    """
    NONE = "none"
    SINGLE = "single"
    MULTIPLE = "multiple"

class OptimizationResult:
    """
    Standardized result object for all algorithms.
    """
    def __init__(self, x, fun: float, algorithm: str = ""):
        """
        :param x: Solution vector (list, array-like)
        :param fun: Objective function value at x
        :param algorithm: Name/identifier of the algorithm used
        """
        self.x = x
        self.fun = float(fun)
        self.algorithm = str(algorithm)

class BaseOptimizer(ABC):
    """
    Abstract base class for optimization algorithms tested with pyGOLD.

    This class defines the interface that all optimization algorithms must
    implement to be compatible with pyGOLD benchmark testing. It enforces a
    standardized approach to handling different algorithm types and their
    initialization requirements.

    Algorithm Types:
    ----------------
    Algorithms must declare one of three initialization types:

    1. NONE: Algorithms that don't use initial conditions
       - Will be run once per problem in benchmarking

    2. SINGLE: Algorithms that require one initial point
       - Will be run multiple times with different initial points in
         benchmarking

    3. MULTIPLE: Algorithms that require multiple initial points
       - Will be run multiple times with different sets of initial points in
         benchmarking
    """

    def __init__(self, initialization_type: InitializationType, **kwargs):
        """
        Initialize the optimizer.

        :param initialization_type: Type of initialization this algorithm
            requires
        :param kwargs: Algorithm-specific parameters
        """
        if not isinstance(initialization_type, InitializationType):
            raise TypeError(f"initialization_type must be an "
                          f"InitializationType enum, got "
                          f"{type(initialization_type)}")

        self.initialization_type = initialization_type
        self.kwargs = kwargs
        self.name = self.__class__.__name__

    @abstractmethod
    def optimize(self, func: Callable, bounds: List[Tuple[float, float]], x0: Optional[Union[np.ndarray, List[np.ndarray]]] = None, **kwargs) -> OptimizationResult:
        """
        Optimize the given function within specified bounds.

        :param func: Objective function to minimize. Should accept a numpy
            array and return a scalar float.
        :type func: Callable
        :param bounds: List of (min, max) tuples specifying the bounds for
            each dimension.
        :type bounds: List[Tuple[float, float]]
        :param x0: Initial condition(s). Type depends on initialization_type:
            - NONE: Should be None (ignored if provided)
            - SINGLE: Single initial point as array-like
            - MULTIPLE: List of initial points as arrays
        :type x0: Optional[Union[np.ndarray, List[np.ndarray]]]
        :param kwargs: Additional algorithm-specific parameters.
        :type kwargs: dict

        :returns: The result of the optimization process.
        :rtype: OptimizationResult
        """
        pass
