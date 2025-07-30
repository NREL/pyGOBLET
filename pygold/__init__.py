"""
pyGOLD: Python Global Optimization Library (for Development)
"""

# Direct access to common functions
from . import problems
from .problems import floris
from .problems import standard
from .default_runners import run_solvers
from .postprocessing import postprocess_data
from .cocopp_interface import configure_testbed
from .problems.standard import get_standard_problems

__all__ = [
    'problems',
    'floris',
    'standard',
    'get_standard_problems',
    'run_solvers',
    'postprocess_data',
    'configure_testbed'
]
