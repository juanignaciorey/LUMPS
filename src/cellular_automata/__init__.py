"""
Cellular Automata Evolution Module
Generates computational lumps through evolutionary processes
"""

from .core import CellularAutomaton
from .evolution import CAEvolver
from .fitness import (
    expand_fitness,
    symmetry_fitness,
    count_fitness,
    topology_fitness,
    replicate_fitness
)
from .task_generator import ARCTaskGenerator

__all__ = [
    'CellularAutomaton',
    'CAEvolver',
    'expand_fitness',
    'symmetry_fitness',
    'count_fitness',
    'topology_fitness',
    'replicate_fitness',
    'ARCTaskGenerator'
]
