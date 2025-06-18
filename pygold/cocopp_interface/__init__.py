"""
COCOPP Interface Module

This module provides integration with COCOPP (COmparing Continuous Optimizers)
for generating performance plots and tables.
"""
from .interface import log_coco_from_results, configure_testbed

__all__ = [
    'log_coco_from_results',
    'configure_testbed'
]
