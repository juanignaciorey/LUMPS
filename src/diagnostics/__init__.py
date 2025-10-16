"""
Diagnostics Module
Meta-learning metrics and visualization tools
"""

from .metrics import (
    compute_transfer_gap,
    measure_lump_diversity,
    test_size_generalization,
    check_primitive_emergence
)
from .visualizer import DiagnosticVisualizer

__all__ = [
    'compute_transfer_gap',
    'measure_lump_diversity',
    'test_size_generalization',
    'check_primitive_emergence',
    'DiagnosticVisualizer'
]

