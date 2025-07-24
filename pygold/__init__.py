"""
pyGOLD: Python Global Optimization Library for Development
"""

# Direct access to common functions
from . import problems
from .problems import floris_problems
from .problems import standard_problems
from .default_runners import run_solvers
from .postprocessing import postprocess_data
from .cocopp_interface import configure_testbed

__all__ = [
    'problems',
    'floris_problems',
    'standard_problems',
    'run_solvers',
    'postprocess_data',
    'configure_testbed'
]
