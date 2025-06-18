"""
pyGOLD: Python Global Optimization Library for Development
"""

# Direct access to common functions
from .default_runners import run_solvers
from .postprocessing import compute_performance_ratios, compute_performance_profiles, plot_performance_profiles
from .cocopp_interface import configure_testbed

__all__ = [
    'run_solvers',
    'compute_performance_ratios',
    'compute_performance_profiles',
    'plot_performance_profiles',
    'configure_testbed'
]
